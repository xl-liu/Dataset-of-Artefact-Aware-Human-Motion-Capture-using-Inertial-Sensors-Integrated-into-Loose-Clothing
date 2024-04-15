#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is application show how to extract the accelerometer, gyroscope and magnetometer
measurements from  the Dataset of
"Towards Artefact Aware Human Motion Capture using Inertial Sensors Integrated into Loose Clothing"
https://zenodo.org/record/5948725


@author: michael lorenz, lorenz@cs.uni-kl.de
"""
import h5py
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import os
from tqdm import tqdm
from articulate.model import ParametricModel
from utils.DataHelpers import dictPerson2Activ
from config import paths, joint_params

def geodesic_angle(a: torch.Tensor, b: torch.Tensor):
    ''' as defined by Zhou et al in their paper
    '''
    # a = a.view(-1, 3, 3)
    # b = b.view(-1, 3, 3)
    c = torch.bmm(a.transpose(1,2), b)
    cos = (c.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1) - 1.) / 2.
    cos = torch.clamp(cos, -1, 1)
    angles = torch.acos(cos)

    return angles 

def quaternion_to_angle_axis(quaternion):
    # Make sure the quaternion is normalized
    quaternion = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)

    # Extract the scalar part (w) and the vector part (x, y, z)
    w = quaternion[..., 0]
    vector_part = quaternion[..., 1:]

    # Compute the angle
    angles = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))

    # Prevent division by zero
    small_angle = angles < 1e-6
    sin_theta_half = torch.sqrt(1 - w * w)
    sin_theta_half[small_angle] = 1.0

    # Compute the axis
    axis = vector_part / sin_theta_half.unsqueeze(-1)

    # Handle the small angles separately
    axis[small_angle] = torch.zeros_like(vector_part[small_angle])

    # Return angle and axis as angle-axis (the magnitude of axis is the angle)
    angle_axis = axis * angles.unsqueeze(-1)
    
    return angle_axis

def quaternion_to_rotation_matrix(quaternion):
    # Ensure the quaternion is normalized
    quaternion = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)

    # Extract components of the quaternion
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    # Prepare components for the matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Compute rotation matrix components
    r11 = 1 - 2 * (yy + zz)
    r12 = 2 * (xy - wz)
    r13 = 2 * (xz + wy)
    r21 = 2 * (xy + wz)
    r22 = 1 - 2 * (xx + zz)
    r23 = 2 * (yz - wx)
    r31 = 2 * (xz - wy)
    r32 = 2 * (yz + wx)
    r33 = 1 - 2 * (xx + yy)

    # Stack components into matrix form
    rotation_matrix = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1).reshape(-1, 3, 3)
    
    return rotation_matrix.reshape(quaternion.shape[0], quaternion.shape[1], 3, 3)

def rotation_matrix_to_axis_angle(r: torch.Tensor):
    """
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result

def q_inv(q):
    ''' assume quaternion of shape n x 4'''
    qi = q.clone()
    qi[:,1:] = -qi[:,1:]
    return qi

def quaternion_product(q1, q2):
    # Extract components
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Calculate the product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack them into a new tensor
    return torch.stack((w, x, y, z), dim=-1)

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def inverse_kinematics_q(parents, global_quats):
    """ inverse kinematics using quaternions as input
    """
    local_quats = torch.zeros_like(global_quats)    # nj x n x 4
    for c, p in enumerate(parents):
        if c == 0:
            local_quats[0] = global_quats[0]
        else:
            local_quats[c] = qmul(q_inv(global_quats[p]), global_quats[c])
    return local_quats

def inverse_kinematics_r(parents, global_rots):
    """ inverse kinematics using rotation matrices as input
    """
    local_rots = torch.zeros_like(global_rots)    # nj x n x 3 x 3
    for c, p in enumerate(parents):
        if c == 0:
            local_rots[0] = global_rots[0]
        else:
            local_rots[c] = torch.matmul(torch.linalg.inv(global_rots[p]), global_rots[c])
    return local_rots

def _find_rotation_offset_torch(v1, v2):
    """
    find the rotation matrix that aligns v2 to v1 in pytorch
    
    parameters:
    - v1: torch tensor of shape (3,), representing the target vector.
    - v2: torch tensor of shape (3,), representing the source vector.
    
    returns:
    - R: The rotation matrix that rotates v2 to align with v1.
    """
    # Ensure inputs are floats for accurate math operations
    v1 = v1.float()
    v2 = v2.float()
    
    # Normalize the input vectors
    v1 = v1 / torch.norm(v1)
    v2 = v2 / torch.norm(v2)
    
    # Find the axis of rotation (cross product of v1 and v2)
    axis = torch.linalg.cross(v1, v2)
    axis_length = torch.norm(axis)
    if axis_length > 1e-6:  # To avoid division by zero
        axis = axis / axis_length
    
    # Find the angle of rotation (dot product of v1 and v2)
    angle = torch.arccos(torch.dot(v1, v2))
    
    # Using Rodrigues' rotation formula to compute the rotation matrix
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
    
    return R

def _align_frame(acc_in, gyro_in, oris):
    """
    transform local IMU readings to the global frame

    parameters: 
    - acc_in: n x 3 torch tensor, accelerometer reading in local sensor frame
    - gyro_in: n x 3 torch tensor, gyroscope reading in local sensor frame
    - oris: n x 3 x 3 torch tensor, orientation in global bone frame

    returns:
    - acc_global: n x 3 torch tensor, accelerometer reading in global bone frame
    - gyro_global: n x 3 torch tensor, gyroscope reading in global bone frame
    """ 
    # take the average acc reading while standing still
    n_window = 20
    a1 = acc_in[:n_window].mean(axis=0)
    r_global = oris[:n_window].transpose(1,2)   # inverse of R
    a2 = r_global.matmul(torch.tensor([0, -1., 0]))
    a2 = a2[:n_window].mean(axis=0)

    # find the offset
    r_offset = _find_rotation_offset_torch(a1, a2)

    # apply on the data to get global 
    acc_bone = r_offset.matmul(acc_in.transpose(0,1)).transpose(0,1)
    acc_global = oris.matmul(acc_bone.unsqueeze(-1)).squeeze(-1)

    gyro_bone = r_offset.matmul(gyro_in.transpose(0,1)).transpose(0,1)
    gyro_global = oris.matmul(gyro_bone.unsqueeze(-1)).squeeze(-1)

    return acc_global, gyro_global

def _rotate_bones(joint_angles, r_init=R):
    """
    transform the joint orientations so that the first frame matches with the desired bone angle 

    parameters:
    - joint_angles: n x 3 x 3 torch tensor, original 3x3 orientations of n frames
    - r_init: scipy Rotation object, reference orientation

    returns:
    - joint_angles_rotated: n x 3 x 3 torch tensor, rotated joint_anlges
    """
    r_t = torch.tensor(r_init.as_matrix()).float()
    r_rot = torch.linalg.inv(joint_angles[0]).matmul(r_t)
    joint_angles_rotated = joint_angles.matmul(r_rot)
    return joint_angles_rotated

def _get_imu_data(h5_filename, test_subject):
    """
    extract IMU data from an h5 file
    based on code from https://github.com/lorenzcsunikl/Dataset-of-Artefact-Aware-Human-Motion-Capture-using-Inertial-Sensors-Integrated-into-Loose-Clothing/blob/main/apps/mainGetIMUData.py
    """    
    h5 = h5py.File(h5_filename,'r')

    #get the order of the joints
    order_abs = [n.decode('utf-8')
                for n in list(h5['P'+str(test_subject)]['orderAbs'][...])]
    
    # select the subset of sensors 
    imu_idx = [order_abs.index(seg) for seg in joint_params.IMU_SEG]

    # get the raw IMU reading
    accs = h5['P'+str(test_subject)]['acc'][...]   # 17 x n x 3
    gyrs = h5['P'+str(test_subject)]['gyr'][...]   # 17 x n x 3
    # mag = getValuesOf(h5, 'mag')[idx_segment_take,:,:]
    acc = torch.from_numpy(accs[imu_idx, :, :]).float().transpose(0,1)      # n x 6 x 3
    gyro = torch.from_numpy(gyrs[imu_idx, :, :]).float().transpose(0,1)
    quats_abs = h5['P'+str(test_subject)]['quatAbs'][...] # absolute orientation 

    h5.close()

    ori = quaternion_to_rotation_matrix(
        torch.from_numpy(quats_abs[imu_idx,:,:]).float())  # 6 x n x 3 x 3
    ori = ori.transpose(0,1)   # n x 6 x 3 x 3

    # rotation offset from the inertial frame to SMPL body frame
    r_it = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).float()  
    r1 = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0.]]).float()

    ori = r_it.matmul(ori)
    ori = r1.matmul(ori).matmul(r1.t())

    # fix the initial reading from the known bone angle
    # head
    # r_head = torch.linalg.inv(ori[0, 4])
    # ori[:,4] = ori[:,4].matmul(r_head)
    
    # left forearm
    r_left_elbow = R.from_euler('zx', [-80, -10], degrees=True)
    ori[:,0] = _rotate_bones(ori[:,0].clone(), r_left_elbow)    
    # right forearm
    r_right_elbow = R.from_euler('zx', [80, -10], degrees=True)
    ori[:,1] = _rotate_bones(ori[:,1].clone(), r_right_elbow)

    acc_global = acc.clone()
    gyro_global = gyro.clone()

    for i in range(6):
        acc_global[:,i], gyro_global[:,i] = _align_frame(acc[:,i], gyro[:,i], ori[:,i])

    # return acc, gyro, ori
    return acc_global, gyro_global, ori

def _npose_to_tpose_0(rots_global):
    """
    convert the xsens resting pose convention (npose) to smpl resting pose convention (tpose) 
    by manually fixing every joint along the arms
    """
    # head
    r_head = torch.linalg.inv(rots_global[2, 0])
    rots_global[2] = rots_global[2].matmul(r_head)

    # right arm tree
    r_right_clavicle = R.from_euler('z', 10, degrees=True)
    rots_global[13] = _rotate_bones(rots_global[13].clone(), r_right_clavicle)
    r_right_shoulder = R.from_euler('z', 78, degrees=True)
    rots_global[14] = _rotate_bones(rots_global[14].clone(), r_right_shoulder)
    r_right_elbow = R.from_euler('zx', [90, -10], degrees=True)
    rots_global[15] = _rotate_bones(rots_global[15].clone(), r_right_elbow)
    r_right_hand = R.from_euler('zx', [90, -10], degrees=True)
    rots_global[16] = _rotate_bones(rots_global[16].clone(), r_right_hand)

    # left arm tree
    r_left_clavicle = R.from_euler('z', -10, degrees=True)
    rots_global[9] = _rotate_bones(rots_global[9].clone(), r_left_clavicle)
    r_left_shoulder = R.from_euler('z', -78, degrees=True)
    rots_global[10] = _rotate_bones(rots_global[10].clone(), r_left_shoulder)
    r_left_elbow = R.from_euler('zx', [-90, -10], degrees=True)
    rots_global[11] = _rotate_bones(rots_global[11].clone(), r_left_elbow)     
    r_left_hand = R.from_euler('zx', [-90, -10], degrees=True)
    rots_global[12] = _rotate_bones(rots_global[12].clone(), r_left_hand)
    
    return rots_global

def _npose_to_tpose_1(rots_global):
    """
    convert the xsens resting pose convention (npose) to smpl resting pose convention (tpose) 
    by aligning with pose readings from a smpl body performing npose
    """
    npose = torch.load(paths.npose_file)
    body_model = ParametricModel(paths.smpl_file, use_pose_blendshape=False)
    npose_global, _ = body_model.forward_kinematics(npose.view(1, 24, 3, 3))
    npose_global = npose_global[0]

    pose_global_mat = rots_global.clone()
    for i in range(24):
        joint_name = joint_params.SMPL_TO_XSENS[joint_params.SMPL_JOINTS[i]]
        if joint_name in joint_params.CORRECTION:   # only correct the upper body
            r_ref = R.from_matrix(npose_global[i].cpu().detach().numpy())
            j = joint_params.REORDER.index(joint_name)
            pose_global_mat[j] = _rotate_bones(rots_global[j], r_ref)     

    return pose_global_mat

def _get_smpl_pose(h5_filename, test_subject):
    """
    convert the orientation readings from the clothing dataset to smpl poses
    """

    h5 = h5py.File(h5_filename,'r')
    # get the order of the joints
    order_abs = [n.decode('utf-8')
                for n in list(h5['P'+str(test_subject)]['orderAbs'][...])]
    # absolute orientation
    quats_abs = h5['P'+str(test_subject)]['quatAbs'][...] 
    h5.close()
    
    # reorder the joints 
    order = [order_abs.index(seg) for seg in joint_params.REORDER]
    quats_global = torch.from_numpy(quats_abs[order,:,:]).float()
    rots_global = quaternion_to_rotation_matrix(quats_global)   # 17 x n x 3 x 3

    # rotation offset from the inertial frame to body frame
    r_it = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).float()  
    r1 = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0.]]).float()

    rots_global= r_it.matmul(rots_global)      # 17 x n x 3 x 3
    rots_global = r1.matmul(rots_global).matmul(r1.t())

    # make sure the root is aligned with the inertial frame
    r_root = rots_global[0,0]
    rots_global = torch.linalg.inv(r_root).matmul(rots_global)

    # npose-tpose convention conversion
    # pos_global_mat = _npose_to_tpose_0(rots_global)
    pose_global_mat = _npose_to_tpose_1(rots_global)

    length = rots_global.shape[1]

    # convert global orientation matrices to local axis angle
    pose_local_aa = torch.zeros(24, length, 3)
    pose_local_mat = inverse_kinematics_r(joint_params.PARENTS, pose_global_mat)   # 17 x n x 3 x 3
    for i in range(24):
        seg_name = joint_params.SMPL_TO_XSENS[joint_params.SMPL_JOINTS[i]]
        if seg_name is not None:
            pose_local_aa[i] = rotation_matrix_to_axis_angle(pose_local_mat[joint_params.REORDER.index(seg_name)])

    pose_local_aa = pose_local_aa.transpose(0,1).reshape(-1, 72)   # n x 72

    return pose_local_aa

def process_(split='test'):

    out_acc, out_ori, out_gyro, out_pose, out_tran = [], [], [], [], []
    out_acc_loose, out_ori_loose, out_gyro_loose = [], [], []
    out_pose_loose = []

    if split == 'train':
        test_subject = [1, 2, 3, 4, 5, 6, 7, 8]
    elif split == 'validation':
        test_subject = [9]
    elif split == 'test':
        test_subject = [10]
    elif split == 'all':
        test_subject = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print(f'---------- {split} split----------')
    for seq in tqdm(joint_params.SEQUENCES):
        for ts in test_subject:
            # read the loose dataset
            filename_loose = paths.clothing_dataset + f'/Loose_{seq}.h5'
            acc_loose, gyro_loose, ori_loose = _get_imu_data(filename_loose, ts)

            # read the tight dataset
            filename_tight = paths.clothing_dataset + f'/Tight_{seq}.h5'
            acc, gyro, ori = _get_imu_data(filename_tight, ts)

            # get the gt pose
            pose = _get_smpl_pose(filename_tight, ts)

            if seq == 'longterm':
                time_activity = dictPerson2Activ[ts]
                beginning = 0
                for (end,act) in time_activity:
                    if act == 'NPose': 
                        continue    # merge the npose frames with the next one
                    out_acc.append(acc[beginning:end])
                    out_gyro.append(gyro[beginning:end])
                    out_ori.append(ori[beginning:end])
                    out_pose.append(pose[beginning:end])
                    out_pose_loose.append(pose[beginning:end])
                    out_tran.append(torch.zeros(acc.shape[0], 3))
                    out_acc_loose.append(acc_loose[beginning:end])
                    out_gyro_loose.append(gyro_loose[beginning:end])
                    out_ori_loose.append(ori_loose[beginning:end])
                    beginning = end
            else:
                out_acc.append(acc)
                out_gyro.append(gyro)
                out_pose.append(pose)
                out_pose_loose.append(pose)
                out_ori.append(ori)
                out_tran.append(torch.zeros(acc.shape[0], 3))
                out_acc_loose.append(acc_loose)
                out_gyro_loose.append(gyro_loose)
                out_ori_loose.append(ori_loose)

    # save to .pt files
    out_filename_tight = os.path.join(paths.tightimu_dir, f'{split}.pt')
    torch.save({'acc': out_acc, 'ori': out_ori, 
                'pose': out_pose, 'tran': out_tran, 
                'gyro': out_gyro,
                },
                out_filename_tight)
    print('Preprocessed data is saved at', out_filename_tight)

    out_filename_loose = os.path.join(paths.looseimu_dir, f'{split}.pt')
    torch.save({'acc': out_acc_loose, 'ori': out_ori_loose, 
                'pose': out_pose_loose, 'tran': out_tran, 
                'gyro': out_gyro_loose,
                },
                out_filename_loose)
    print('Preprocessed data is saved at', out_filename_loose)


if __name__ == '__main__':

    process_(split='test')
    process_(split='train')
    process_(split='validation')

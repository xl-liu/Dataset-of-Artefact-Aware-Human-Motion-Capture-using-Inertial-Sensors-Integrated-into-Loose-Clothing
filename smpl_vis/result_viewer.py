import torch
import pickle as pkl
import torchaudio.functional as F

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from scipy.signal import savgol_filter, firwin
from scipy import signal
import numpy as np
import math
from aitviewer.renderables.meshes import Meshes
from config import paths

SEQ = 5
START_IDX = 30 
END_IDX = 4500

def rotation_matrix_to_angle_axis(rotation_matrix):
    # Handle the case where the rotation is very close to 0
    # Set a small threshold
    epsilon = 1e-6

    # Calculate the angle
    trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, min=-1.0, max=1.0))

    # Create a mask for the edge case where theta is very small (close to 0 or 2*pi)
    small_angle = theta.abs() < epsilon

    # Calculate each element of the axis vector
    u_x = rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2]
    u_y = rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0]
    u_z = rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]

    # Stack and normalize
    u = torch.stack([u_x, u_y, u_z], dim=-1) / (2 * torch.sin(theta).unsqueeze(-1))

    # Handle small angles separately
    u[small_angle] = torch.zeros_like(u[small_angle])

    # Multiply the axis by the angle to get the angle-axis representation
    angle_axis = u * theta.unsqueeze(-1)

    return angle_axis

def avg_mesh(mesh_data, n_window):
    ''' mesh_data is numpy array of size n x 6890 x 3
    '''
    filtered_mesh_data = mesh_data.copy()
    for i in range(1, n_window):
        filtered_mesh_data[:-i] += mesh_data[i:]
    
    filtered_mesh_data /= n_window

    filtered_mesh_data = np.vstack((filtered_mesh_data[:n_window], filtered_mesh_data[:-n_window]))

    return filtered_mesh_data

def error_map(mesh_gt, mesh_est):
    ''' calculate the mesh color from rmse between the ground truth vertices and esitmated vertices
        return mesh color of shape n x v x 4
    '''
    error = np.linalg.norm(np.sqrt((mesh_gt - mesh_est)**2), axis=2)
    error = 1. - error / np.max(error)
    n = np.shape(mesh_gt)[0]
    v = 6890
    color = np.stack((np.ones((n, v)), error, error, np.ones((n, v))), axis=2)

    return color

if __name__ == "__main__":

    # load the gt pose
    data_gt = torch.load(paths.tightimu_dir + '/test.pt')
    # END_IDX = data_gt['pose'][SEQ].shape[0]
    poses_gt = data_gt['pose'][SEQ][START_IDX:END_IDX]
    oris = data_gt['ori'][SEQ][START_IDX:END_IDX]
    
    # load the results
    # tight
    predictions = torch.load(f'{paths.result_dir}/tight/PIP/{SEQ}.pt')
    poses_in = predictions[0][START_IDX:END_IDX]   # n x 24 x 3 x 3
    poses_t = rotation_matrix_to_angle_axis(poses_in) # n x 24 x 3
    poses_t = poses_t.reshape(-1, 72)

    # loose
    predictions = torch.load(f'{paths.result_dir}/loose/PIP/{SEQ}.pt')
    poses_in = predictions[0][START_IDX:END_IDX]   # n x 24 x 3 x 3
    poses_l = rotation_matrix_to_angle_axis(poses_in) # n x 24 x 3
    poses_l = poses_l.reshape(-1, 72)

    # filtered 
    predictions = torch.load(f'{paths.result_dir}/filtered/PIP/{SEQ}.pt')
    poses_in = predictions[0][START_IDX:END_IDX]   # n x 24 x 3 x 3
    poses_fl = rotation_matrix_to_angle_axis(poses_in) # n x 24 x 3
    poses_fl = poses_fl.reshape(-1, 72)

    # assume the mean shape
    length = poses_gt.shape[0]
    gender = "male"
    betas = torch.zeros((length, 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)
    
    d_offset = torch.tensor([0.5, 0, 0.]).tile((length, 1))     # distance between smpl bodies
    
    alpha=1.0   # transparency

    # create smpl sequences
    smpl_seq_gt = SMPLSequence(poses_body=poses_gt[:, 3:], smpl_layer=smpl_layer, poses_root=poses_gt[:, :3],
                              color=(125 / 255, 125 / 255, 125 / 255, alpha), is_rigged=False,
                              betas=betas)
    
    smpl_seq_t = SMPLSequence(poses_body=poses_t[:, 3:], smpl_layer=smpl_layer, poses_root=poses_t[:, :3],
                              color=(50 / 255, 255 / 255, 125 / 255, alpha), is_rigged=False,
                              betas=betas)

    smpl_seq_l = SMPLSequence(poses_body=poses_l[:, 3:], smpl_layer=smpl_layer, poses_root=poses_l[:, :3],
                              color=(255 / 255, 80 / 255, 50 / 255, alpha), is_rigged=False,
                              betas=betas)
    
    smpl_seq_fl = SMPLSequence(poses_body=poses_fl[:, 3:], smpl_layer=smpl_layer, poses_root=poses_fl[:, :3],
                              color=(75 / 255, 125 / 255, 200 / 255, alpha), is_rigged=False,
                              betas=betas)
    
    # get the vertex positions and calcualte the rmse vertex error
    vertices_t = smpl_seq_t.mesh_seq.vertices # n x 6890 x 3
    vertices_l = smpl_seq_l.mesh_seq.vertices
    vertices_fl = smpl_seq_fl.mesh_seq.vertices
    gt_mesh = smpl_seq_gt.mesh_seq.vertices

    color_map_t = error_map(gt_mesh, vertices_t)
    color_map_l = error_map(gt_mesh, vertices_l)
    color_map_fl = error_map(gt_mesh, vertices_fl)

    mesh0 = Meshes(vertices=gt_mesh, faces=smpl_seq_gt.mesh_seq.faces,
                   vertex_colors=np.ones(color_map_t.shape), name="gt", position=-3*d_offset)   
    
    mesh1 = Meshes(vertices=vertices_t, faces=smpl_seq_t.mesh_seq.faces, 
                   vertex_colors=color_map_t, name="tight", position=-d_offset)
    
    mesh2 = Meshes(vertices=vertices_l, faces=smpl_seq_l.mesh_seq.faces, 
                   vertex_colors=color_map_l, name="loose", position=3*d_offset)
    
    mesh3 = Meshes(vertices=vertices_fl, faces=smpl_seq_fl.mesh_seq.faces, 
                   vertex_colors=color_map_fl, name="filtered", position=d_offset)

    # # plot the IMUs
    # # find the joint positions to attach the IMUs
    # _, joints = smpl_layer(
    #     poses_body=poses_gt[:, 3:].float().to(C.device),
    #     poses_root=poses_gt[:, :3].float().to(C.device),
    #     betas=betas,
    # )
    # sensor_idxs = [20, 21, 4, 5, 15, 0]
    # sensor_pos = joints[:, sensor_idxs]
    # sensor_pos[:,:,0] -= 0.5
    # rbs = RigidBodies(sensor_pos.cpu().numpy(), oris.cpu().numpy())

    # Add everything to the scene
    v = Viewer()
    v.playback_fps = 60.0
    v.scene.light_mode = 'dark'
    v.scene.floor.enabled = False
    v.scene.origin.enabled = False
    v.scene.add(mesh0, mesh1, mesh2, mesh3)
    v.run()
    
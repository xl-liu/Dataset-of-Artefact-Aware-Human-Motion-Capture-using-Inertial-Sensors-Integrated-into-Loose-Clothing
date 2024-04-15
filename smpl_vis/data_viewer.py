import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

from config import paths

SEQ = 1 # sequence index 
START_IDX = 100     # starting frame
END_IDX = 500      # last frame 

if __name__ == "__main__":

    # load the tight data
    data_t = torch.load(paths.tightimu_dir + '/test.pt')
    poses = data_t['pose'][SEQ][START_IDX:END_IDX]
    oris_t = data_t['ori'][SEQ][START_IDX:END_IDX]

    # load the loose data
    data_l = torch.load(paths.looseimu_dir + '/test.pt')
    oris_l = data_l['ori'][SEQ][START_IDX:END_IDX]

    # downsample to 30 Hz
    poses = poses[::2]
    oris_t = oris_t[::2]
    oris_l = oris_l[::2]

    # assume the mean shape
    gender = "male"
    betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=poses[:, 3:].float().to(C.device),
        poses_root=poses[:, :3].float().to(C.device),
        betas=betas,
    )

    # SMPL joint indices for the 6 IMUs
    sensor_idxs = [20, 21, 4, 5, 15, 0]
    imu_t = RigidBodies(joints[:, sensor_idxs].cpu().numpy(), oris_t.cpu().numpy())
    imu_l = RigidBodies(joints[:, sensor_idxs].cpu().numpy(), oris_l.cpu().numpy())

    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3])
    smpl_seq.mesh_seq.color = smpl_seq.mesh_seq.color[:3] + (0.5,)

    # Add everything to the scene and display at 30 fps
    v = Viewer()
    v.playback_fps = 30.0

    v.scene.add(smpl_seq, imu_t)    # visualize the orientation of the tight imus
    # v.scene.add(smpl_seq, imu_l)  # visualize the orientation of the loose imus
    v.run()
    
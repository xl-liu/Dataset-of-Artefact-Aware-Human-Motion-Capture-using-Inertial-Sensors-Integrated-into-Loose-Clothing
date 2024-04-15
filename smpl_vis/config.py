class paths:
    result_dir = 'data/result'                      # output directory for the evaluation results

    smpl_file = 'models/SMPL_male.pkl'              # official SMPL model path
    physics_model_file = 'models/physics.urdf'      # physics body model path
    plane_file = 'models/plane.urdf'                # (for debug) path to plane.urdf    Please put plane.obj next to it.
    weights_file = 'data/weights.pt'                # network weight file
    physics_parameter_file = 'physics_parameters.json'   # physics hyperparameters

    looseimu_dir = '../data/clothing/loose'
    tightimu_dir = '../data/clothing/tight'
    filteredimu_dir = '../data/clothing/filtered'
    clothing_dataset = '../data/clothing'

    npose_file = 'npose.pt'

class joint_params:
    # skeleton segments used the loose clothing dataset
    # ABS_SEGMENTS = ['L-Forearm', 'L-Upperarm', 'L-Shoulder', 'R-Forearm', 'R-Upperarm', 'R-Shoulder', 'Sternum', 
    #             'Pelvis', 'L-Tigh', 'L-Shank', 'R-Tigh', 'R-Shank', 'Head', 'L-Hand', 'R-Hand', 'L-Foot', 'R-Foot']

    SEQUENCES = ['shoulder_abduction', 'shoulder_flexion', 'squat', 'longterm']

    IMU_SEG = ['L-Forearm','R-Forearm','L-Shank','R-Shank', 'Head', 'Pelvis'] 

    REORDER = ['Pelvis', 'Sternum', 'Head', 
                'L-Tigh', 'L-Shank', 'L-Foot',
                'R-Tigh', 'R-Shank', 'R-Foot',
                'L-Shoulder', 'L-Upperarm', 'L-Forearm', 'L-Hand', 
                'R-Shoulder', 'R-Upperarm', 'R-Forearm', 'R-Hand']

    # index of the parent joints
    PARENTS = [None, 0, 1, 0, 3, 4, 0, 6, 7, 1, 9, 10, 11, 1, 13, 14, 15]

    # joints that need npose->tpose correction
    CORRECTION = ['Pelvis', 'Sternum', 'Head', 
                'L-Shoulder', 'L-Upperarm', 'L-Forearm', 'L-Hand', 
                'R-Shoulder', 'R-Upperarm', 'R-Forearm', 'R-Hand']

    SMPL_JOINTS = ['pelvis', 'l_hip', 'r_hip', 'spine_1', 'l_knee', 'r_knee', 'spine_2', 'l_ankle', 
                'r_ankle', 'spine_3', 'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 
                'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']

    SMPL_TO_XSENS = {'pelvis': 'Pelvis', 
                'l_hip': 'L-Tigh', 
                'r_hip': 'R-Tigh', 
                'spine_1': 'Sternum', 
                'l_knee': 'L-Shank', 
                'r_knee': 'R-Shank', 
                'spine_2': None, 
                'l_ankle': 'L-Foot', 
                'r_ankle': 'R-Foot', 
                'spine_3': None, 
                'l_foot': None, 
                'r_foot': None, 
                'neck': 'Head', 
                'l_collar': 'L-Shoulder', 
                'r_collar': 'R-Shoulder', 
                'head': None, 
                'l_shoulder': 'L-Upperarm', 
                'r_shoulder': 'R-Upperarm', 
                'l_elbow': 'L-Forearm', 
                'r_elbow': 'R-Forearm', 
                'l_wrist': 'L-Hand', 
                'r_wrist': 'R-Hand', 
                'l_hand': None, 
                'r_hand': None}
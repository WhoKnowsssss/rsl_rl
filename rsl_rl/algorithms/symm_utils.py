import torch

# BODY_NAMES = [
#     "pelvis",
#     "left_hip_pitch_link",
#     "right_hip_pitch_link",
#     "waist_yaw_link",
#     "left_hip_roll_link",
#     "right_hip_roll_link",
#     "waist_roll_link",
#     "left_hip_yaw_link",
#     "right_hip_yaw_link",
#     "torso_link",
#     "left_knee_link",
#     "right_knee_link",
#     "left_shoulder_pitch_link",
#     "right_shoulder_pitch_link",
#     "left_ankle_pitch_link",
#     "right_ankle_pitch_link",
#     "left_shoulder_roll_link",
#     "right_shoulder_roll_link",
#     "left_ankle_roll_link",
#     "right_ankle_roll_link",
#     "left_shoulder_yaw_link",
#     "right_shoulder_yaw_link",
#     "left_elbow_link",
#     "right_elbow_link",
#     "left_wrist_roll_link",
#     "right_wrist_roll_link",
#     "left_wrist_pitch_link",
#     "right_wrist_pitch_link",
#     "left_wrist_yaw_link",
#     "right_wrist_yaw_link",
# ]

BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

def get_joint_permutation_matrix(joint_names):
    # Create two lists, one for left body parts and one for right body parts
    left_names = [name for name in joint_names if name.startswith('left')]
    right_names = [name for name in joint_names if name.startswith('right')]
    
    # Check if the number of left and right body parts are equal
    if len(left_names) != len(right_names):
        raise ValueError("The number of left and right body parts must be equal for permutation.")
    
    matrix = torch.arange(len(joint_names))

    # Create a mapping from left body parts to right body parts
    for i, left_name in enumerate(left_names):
        left_index = joint_names.index(left_name)
        right_name = 'right' + left_name.split('left')[1]  # Replace 'L_' with 'R_'
        
        if right_name in joint_names:
            right_index = joint_names.index(right_name)
            matrix[left_index] = right_index
            matrix[right_index] = left_index
    return matrix

def get_joint_reflection_matrix(joint_names):
    matrix = torch.ones(len(joint_names))

    # Create a mapping from left body parts to right body parts
    for i, name in enumerate(joint_names):
        if 'roll' in name or 'yaw' in name:
            matrix[i] = -1
    return matrix

def get_body_permutation_matrix(body_names):
    # Create two lists, one for left body parts and one for right body parts
    left_names = [name for name in body_names if name.startswith('left')]
    right_names = [name for name in body_names if name.startswith('right')]
    
    # Check if the number of left and right body parts are equal
    if len(left_names) != len(right_names):
        raise ValueError("The number of left and right body parts must be equal for permutation.")
    
    matrix = torch.arange(len(body_names))

    # Create a mapping from left body parts to right body parts
    for i, left_name in enumerate(left_names):
        left_index = body_names.index(left_name)
        right_name = 'right' + left_name.split('left')[1]  # Replace 'L_' with 'R_'
        
        if right_name in body_names:
            right_index = body_names.index(right_name)
            matrix[left_index] = right_index
            matrix[right_index] = left_index
    return matrix

def get_reflect_op(reps):
        reps_shape = []
        for i in range(len(reps)):
            assert reps[i].shape[1] == reps[i].shape[0]
            reps_shape.append(reps[i].shape[0])

        reflect_op = torch.zeros((sum(reps_shape), sum(reps_shape)))

        for i in range(len(reps)):
            idx0 = sum(reps_shape[:i])
            idx1 = sum(reps_shape[:i + 1])
            reflect_op[idx0:idx1, idx0:idx1] = reps[i]

        return reflect_op

def get_reflect_reps(body_names, joint_names):
    Rd = torch.eye(3)
    Rd[1, 1] = -1
    Rd_pseudo = torch.eye(3)
    Rd_pseudo[[0, 2], [0, 2]] = -1

    jperm = get_joint_permutation_matrix(joint_names)
    jref = get_joint_reflection_matrix(joint_names)
    Q = torch.zeros((len(jperm), len(jperm)))
    Q[torch.arange(len(jperm)), jperm] = jref

    bperm = get_body_permutation_matrix(body_names)
    Q_Rd = torch.zeros((len(bperm), len(bperm), 3, 3))
    Q_Rd[torch.arange(len(bperm)), bperm] = Rd[None,:,:].repeat(len(bperm), 1, 1)
    Q_Rd = Q_Rd.permute(0,2,1,3).reshape(len(bperm)*3,len(bperm)*3)
    Q_Rd_pseudo = torch.zeros((len(bperm), len(bperm), 3, 3))
    Q_Rd_pseudo[torch.arange(len(bperm)), bperm] = Rd_pseudo[None,:,:].repeat(len(bperm), 1, 1)
    Q_Rd_pseudo = Q_Rd_pseudo.permute(0,2,1,3).reshape(len(bperm)*3,len(bperm)*3)

    Q_Rd_pseudo_rot6d = torch.zeros((len(bperm), len(bperm), 6, 6))
    Rd_pseudo_6d = torch.eye(6)
    Rd_pseudo_6d[[1, 3, 5], [1, 3, 5]] = -1
    Q_Rd_pseudo_rot6d[torch.arange(len(bperm)), bperm] = Rd_pseudo_6d[None,:,:].repeat(len(bperm), 1, 1)
    Q_Rd_pseudo_rot6d = Q_Rd_pseudo_rot6d.permute(0,2,1,3).reshape(len(bperm)*6,len(bperm)*6)
    return Q, Rd, Rd_pseudo, Q_Rd, Q_Rd_pseudo, Q_Rd_pseudo_rot6d, len(bperm)


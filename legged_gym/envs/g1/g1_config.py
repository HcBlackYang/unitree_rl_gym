from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #    'left_hip_yaw_joint' : 0. ,
        #    'left_hip_roll_joint' : 0,
        #    'left_hip_pitch_joint' : -0.1,
        #    'left_knee_joint' : 0.3,
        #    'left_ankle_pitch_joint' : -0.2,
        #    'left_ankle_roll_joint' : 0,
        #    'right_hip_yaw_joint' : 0.,
        #    'right_hip_roll_joint' : 0,
        #    'right_hip_pitch_joint' : -0.1,
        #    'right_knee_joint' : 0.3,
        #    'right_ankle_pitch_joint': -0.2,
        #    'right_ankle_roll_joint' : 0,
        #    'torso_joint' : 0.
        # }

        default_joint_angles = {
            # 腿部（共 12 个关节）
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            # 躯干（3 个关节）
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # 左臂（7 个关节）
            "left_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.1,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.5,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            # 右臂（7 个关节）
            "right_shoulder_pitch_joint": 0.3,
            "right_shoulder_roll_joint": -0.1,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.5,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            # 左手（7 个关节），若不控制手部，可全部设为固定值
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_index_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            # 右手（7 个关节）
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_index_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
        }
    
    class env(LeggedRobotCfg.env):
        # num_observations = 47
        # num_privileged_obs = 50
        # num_actions = 12

        num_observations = 140
        num_privileged_obs = 143
        num_actions = 43


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        # stiffness = {'hip_yaw': 100,
        #              'hip_roll': 100,
        #              'hip_pitch': 100,
        #              'knee': 150,
        #              'ankle': 40,
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 2,
        #              'hip_roll': 2,
        #              'hip_pitch': 2,
        #              'knee': 4,
        #              'ankle': 2,
        #              }  # [N*m/rad]  # [N*m*s/rad]

        stiffness = {
            # 腿部
            "hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40,
            # 躯干
            "waist": 30,
            # 手臂
            "shoulder": 40, "elbow": 25, "wrist": 15,
            # 手部（如果希望固定手部，可将这些值设为 0）
            "thumb": 0, "middle": 0, "index": 0,
        }
        damping = {
            # 腿部
            "hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2,
            # 躯干
            "waist": 1,
            # 手臂
            "shoulder": 1.5, "elbow": 1, "wrist": 0.5,
            # 手部
            "thumb": 0, "middle": 0, "index": 0,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        file = '/home/blake/g1_gym/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        # penalize_contacts_on = ["hip", "knee"]
        penalize_contacts_on = ["hip", "knee", "shoulder", "elbow", "waist"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

  

# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
#
#
# class G1KitchenRoughCfg(LeggedRobotCfg):
#
#     class env(LeggedRobotCfg.env):
#         num_envs = 4096
#         # 移除硬编码的num_observations和num_actions
#         # 这些值应该在环境初始化时从实际机器人动态获取
#         num_observations = None  # 将在环境初始化时设置
#         num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
#         num_actions = None  # 将在环境初始化时设置为self.num_dof
#         env_spacing = 3.  # not used with heightfields/trimeshes
#         send_timeouts = True  # send time out information to the algorithm
#         episode_length_s = 20  # episode length in seconds
#         test = False
#
#     class init_state(LeggedRobotCfg.init_state):
#         pos = [5.0, 5.0, 0.8]  # x,y,z [m]
#         default_joint_angles = {  # = target angles [rad] when action = 0.0
#             'left_hip_yaw_joint': 0.,
#             'left_hip_roll_joint': 0,
#             'left_hip_pitch_joint': -0.1,
#             'left_knee_joint': 0.3,
#             'left_ankle_pitch_joint': -0.2,
#             'left_ankle_roll_joint': 0,
#             'right_hip_yaw_joint': 0.,
#             'right_hip_roll_joint': 0,
#             'right_hip_pitch_joint': -0.1,
#             'right_knee_joint': 0.3,
#             'right_ankle_pitch_joint': -0.2,
#             'right_ankle_roll_joint': 0,
#             'torso_joint': 0.
#         }
#
#     class env(LeggedRobotCfg.env):
#         num_observations = 47
#         num_privileged_obs = 50
#         num_actions = 12
#         rollout_horizon = 200
#
#     class domain_rand(LeggedRobotCfg.domain_rand):
#         randomize_friction = True
#         friction_range = [0.1, 1.25]
#         randomize_base_mass = True
#         added_mass_range = [-1., 3.]
#         push_robots = True
#         push_interval_s = 5
#         max_push_vel_xy = 1.5
#
#     class control(LeggedRobotCfg.control):
#         # PD Drive parameters:
#         control_type = 'P'
#         # PD Drive parameters:
#         stiffness = {'hip_yaw': 100,
#                      'hip_roll': 100,
#                      'hip_pitch': 100,
#                      'knee': 150,
#                      'ankle': 40,
#                      }  # [N*m/rad]
#         damping = {'hip_yaw': 2,
#                    'hip_roll': 2,
#                    'hip_pitch': 2,
#                    'knee': 4,
#                    'ankle': 2,
#                    }  # [N*m/rad]  # [N*m*s/rad]
#         # action scale: target angle = actionScale * action + defaultAngle
#         action_scale = 0.25
#         # decimation: Number of control action updates @ sim DT per policy DT
#         decimation = 4
#
#     class asset(LeggedRobotCfg.asset):
#         # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
#         file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
#         name = "g1"
#         foot_name = "ankle_roll"
#         penalize_contacts_on = ["hip", "knee"]
#         terminate_after_contacts_on = ["pelvis"]
#         self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
#         flip_visual_attachments = False
#
#     class rewards(LeggedRobotCfg.rewards):
#         soft_dof_pos_limit = 0.9
#         base_height_target = 0.78
#
#         class scales(LeggedRobotCfg.rewards.scales):
#             tracking_lin_vel = 1.0
#             tracking_ang_vel = 0.5
#             lin_vel_z = -2.0
#             ang_vel_xy = -0.05
#             orientation = -1.0
#             base_height = -10.0
#             dof_acc = -2.5e-7
#             dof_vel = -1e-3
#             feet_air_time = 0.0
#             collision = 0.0
#             action_rate = -0.01
#             dof_pos_limits = -5.0
#             alive = 0.15
#             hip_pos = -1.0
#             contact_no_vel = -0.2
#             feet_swing_height = -20.0
#             contact = 0.18
#
#
# class G1KitchenRoughCfgPPO(LeggedRobotCfgPPO):
#     class policy:
#         init_noise_std = 0.8
#         # 增加网络宽度，提高表现能力
#         actor_hidden_dims = [512, 256, 128]  # 改为更大的网络
#         critic_hidden_dims = [512, 256, 128]  # 改为更大的网络
#         activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
#         # only for 'ActorCriticRecurrent':
#         rnn_type = 'lstm'
#         rnn_hidden_size = 128  # 增加RNN大小
#         rnn_num_layers = 1
#
#         # 保留这一行，确保PPO模型会从环境获取正确的动作维度
#         action_dim_from_env = True
#
#     class algorithm(LeggedRobotCfgPPO.algorithm):
#         entropy_coef = 0.01
#
#         # 关键修改：设置num_mini_batches = 1，避免批次划分问题
#         num_learning_epochs = 5
#         num_mini_batches = 24  # 改为1，不再进行mini-batch划分
#         learning_rate = 1e-4
#
#     class runner(LeggedRobotCfgPPO.runner):
#         policy_class_name = "ActorCriticRecurrent"
#         algorithm_class_name = "PPO"
#         max_iterations = 10000
#         num_steps_per_env = 48  # 尝试一个能被num_mini_batches整除的值
#         save_interval = 50
#         experiment_name = 'g1'
#         run_name = ''


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1KitchenRoughCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # 移除硬编码的num_observations和num_actions
        # 这些值应该在环境初始化时从实际机器人动态获取
        num_observations = None  # 将在环境初始化时设置
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = None  # 将在环境初始化时设置为self.num_dof
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        test = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [5.0, 5.0, 0.8]  # x,y,z [m]
        default_joint_angles = {
            # 腿部
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0,
            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0,

            # 腰部（可锁定）
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,

            # 手臂
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,

            # 新增：Panda手指关节
            'left_panda_finger_joint1': 0.02,
            'left_panda_finger_joint2': 0.02,  # 添加这一行
            'right_panda_finger_joint1': 0.02,
            'right_panda_finger_joint2': 0.02,  # 添加这一行
        }

    class env(LeggedRobotCfg.env):
        num_observations = 104  # 原92 + 4个观测 (左右手各两个手指位置和速度)
        num_privileged_obs = 107  # 原95 + 4个特权观测
        num_actions = 31  # 原29 + 2个手指控制 (左右手各一个独立控制的手指)

        rollout_horizon = 200

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'hip_yaw': 100,
            'hip_roll': 100,
            'hip_pitch': 100,
            'knee': 150,
            'ankle': 40,
            'waist_yaw': 30,
            'waist_roll': 30,
            'waist_pitch': 30,
            'shoulder': 40,
            'elbow': 25,
            'wrist': 15,
            'finger': 20,  # 添加手指关节刚度
        }
        damping = {
            'hip_yaw': 2,
            'hip_roll': 2,
            'hip_pitch': 2,
            'knee': 4,
            'ankle': 2,
            'waist_yaw': 1,
            'waist_roll': 1,
            'waist_pitch': 1,
            'shoulder': 1,
            'elbow': 1,
            'wrist': 0.5,
            'finger': 0.5,  # 添加手指关节阻尼
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        # 更新为您的新URDF文件
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_panda.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales(LeggedRobotCfg.rewards.scales):
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


class G1KitchenRoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        # 增加网络宽度，提高表现能力
        actor_hidden_dims = [512, 256, 128]  # 改为更大的网络
        critic_hidden_dims = [512, 256, 128]  # 改为更大的网络
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 128  # 增加RNN大小
        rnn_num_layers = 1

        # 保留这一行，确保PPO模型会从环境获取正确的动作维度
        action_dim_from_env = True

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

        # 关键修改：设置num_mini_batches = 1，避免批次划分问题
        num_learning_epochs = 5
        num_mini_batches = 24  # 改为1，不再进行mini-batch划分
        learning_rate = 1e-4

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        algorithm_class_name = "PPO"
        max_iterations = 10000
        num_steps_per_env = 48  # 尝试一个能被num_mini_batches整除的值
        save_interval = 50
        experiment_name = 'g1'
        run_name = ''

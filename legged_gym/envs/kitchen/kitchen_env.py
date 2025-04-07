
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import os
import math

from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.envs.kitchen.kitchen_utils import parse_lisdf

from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.math import wrap_to_pi
from isaacgym.torch_utils import quat_apply


class G1KitchenRobot(LeggedRobot):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 初始化kitchen actors存储
        self.kitchen_assets = {}
        self.kitchen_poses = {}
        self.kitchen_actors_by_env = {}

        # 调用父类初始化 - 这会创建环境和机器人
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # 重要：确保num_actions与机器人自由度一致
        self.num_actions = self.num_dof
        # 更新配置中的值，确保PPO使用正确的维度
        self.cfg.env.num_actions = self.num_dof

    def create_sim(self):
        """ 重写create_sim方法，修改加载顺序 """
        sim_params = self.sim_params
        if not hasattr(sim_params, "physx"):
            sim_params.physx = gymapi.PhysXParams()

        # # 增加碰撞内存配置
        # sim_params.physx.found_lost_aggregate_pairs_capacity = 52000000  # 略大于报错中的要求

        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, sim_params)

        self._create_ground_plane()
        self._load_kitchen_assets()  # 先加载kitchen资产
        self._create_envs()  # 然后创建环境（包括机器人和kitchen）


    def _load_kitchen_assets(self):
        """加载所有Kitchen资产，在创建环境之前"""
        print("🔍 开始加载Kitchen资产...")
        asset_root = "/home/blake/kitchen-worlds/assets/models/"
        lisdf_path = "/home/blake/kitchen-worlds/assets/scenes/kitchen_basics.lisdf"
        pose_data = parse_lisdf(lisdf_path)

        # 加载所有URDF资产
        for urdf_path, data in pose_data.items():
            urdf_relative_path = os.path.relpath(urdf_path, asset_root)
            if not os.path.exists(urdf_path):
                print(f"⚠️ Warning: URDF 文件不存在: {urdf_path}")
                continue

            pose = data["pose"]
            scale = data["scale"]

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = False
            asset_options.use_mesh_materials = True
            asset_options.override_com = True
            asset_options.override_inertia = True

            # 修改碰撞生成选项，避免凸包分解错误
            asset_options.convex_decomposition_from_submeshes = False  # 禁用从子网格创建凸包
            asset_options.vhacd_enabled = False  # 禁用VHACD
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.01  # 增加厚度可能有助于稳定性

            try:
                object_asset = self.gym.load_asset(self.sim, asset_root, urdf_relative_path, asset_options)
                if object_asset is None:
                    print(f"❌ ERROR: 无法加载 URDF: {urdf_relative_path}")
                    continue

                self.kitchen_assets[urdf_relative_path] = object_asset
                self.kitchen_poses[urdf_relative_path] = pose
                # 存储scale信息
                self.kitchen_scales = getattr(self, 'kitchen_scales', {})
                self.kitchen_scales[urdf_relative_path] = scale
                print(f"✅ 成功加载: {urdf_relative_path} (Scale: {scale})")
            except Exception as e:
                print(f"❌ 加载'{urdf_relative_path}'时发生错误: {e}")
                # 如果加载失败，尝试使用更简单的碰撞设置
                try:
                    print(f"🔄 尝试使用简化选项重新加载'{urdf_relative_path}'")
                    asset_options.convex_decomposition_from_submeshes = False
                    asset_options.vhacd_enabled = False
                    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                    asset_options.use_mesh_materials = False
                    # 使用非常简单的碰撞模型
                    asset_options.create_convex_meshes = False
                    asset_options.replace_cylinder_with_capsule = True

                    object_asset = self.gym.load_asset(self.sim, asset_root, urdf_relative_path, asset_options)
                    if object_asset is not None:
                        self.kitchen_assets[urdf_relative_path] = object_asset
                        self.kitchen_poses[urdf_relative_path] = pose
                        # 存储scale信息
                        self.kitchen_scales = getattr(self, 'kitchen_scales', {})
                        self.kitchen_scales[urdf_relative_path] = scale
                        print(f"✅ 成功使用简化选项加载: {urdf_relative_path} (Scale: {scale})")
                except Exception as e2:
                    print(f"❌ 简化加载仍然失败: {e2}")
                    continue


    def _create_envs(self):
        """重写环境创建方法，创建机器人和kitchen，并处理凸包网格问题"""
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        # 设置机器人动作空间大小
        self.num_actions = self.num_dof

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # 保存body和DOF名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()

        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        # 初始化kitchen actors存储
        self.kitchen_actors_by_env = [[] for _ in range(self.num_envs)]

        spacing_factor = 5.0  # 固定的环境间距放大因子

        # 为每个环境创建机器人和Kitchen组件
        for i in range(self.num_envs):
            # 创建环境（关闭自动网格分布：最后一个参数设为1）
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, 1)
            # 使用环境原点乘以 spacing_factor作为环境偏移（不加随机偏移）
            pos = self.env_origins[i].clone() * spacing_factor
            start_pose.p = gymapi.Vec3(*pos)

            # 创建机器人
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_actor = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_actor, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_actor)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_actor, body_props, recomputeInertia=True)

            self.envs.append(env_handle)
            self.actor_handles.append(robot_actor)

            # 加载所有Kitchen组件，不做判断和异常捕获
            successful_kitchen_actors = []
            for urdf_path, asset in self.kitchen_assets.items():
                self._add_kitchen_actor(i, env_handle, urdf_path, asset, successful_kitchen_actors)

            self.kitchen_actors_by_env[i] = successful_kitchen_actors

        # 设置feet索引和碰撞索引
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

        # 一次性准备模拟 - 在所有actor创建后
        self.gym.prepare_sim(self.sim)
        print("✅ 所有环境和Kitchen场景创建完成！")

    def _add_kitchen_actor(self, env_idx, env_handle, urdf_path, asset, successful_actors_list):
        pose = self.kitchen_poses[urdf_path]
        scale = self.kitchen_scales.get(urdf_path)

        # 确保使用原有的环境原点计算，并应用放大因子
        spacing_factor = 5.0  # 固定的环境间距放大因子
        env_origin = self.env_origins[env_idx].clone() * spacing_factor

        # 打印调试信息
        print(f"环境 {env_idx}: 原点 = {env_origin}, 放大因子 = {spacing_factor}")

        # 定义厨房整体的基准偏移
        kitchen_base_offset = torch.tensor([2.3, 4.7, -0.002], device=self.device)

        # 计算组件的相对位置：保留原有计算方式，但添加调试输出
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(
            float(pose.pos[0]) - kitchen_base_offset[0] + env_origin[0],
            float(pose.pos[1]) - kitchen_base_offset[1] + env_origin[1],
            float(pose.pos[2]) - kitchen_base_offset[2] + env_origin[2]
        )
        transform.r = gymapi.Quat(
            float(pose.quat_wxyz[1]),
            float(pose.quat_wxyz[2]),
            float(pose.quat_wxyz[3]),
            float(pose.quat_wxyz[0])
        )

        # 打印组件最终位置
        print(f"环境 {env_idx}: 组件 {urdf_path} 最终位置 = ({transform.p.x}, {transform.p.y}, {transform.p.z})")

        # 创建厨房组件actor
        kitchen_actor = self.gym.create_actor(
            env_handle,
            asset,
            transform,
            f"kitchen_{urdf_path}",
            env_idx,
            2,  # 碰撞组
            1  # 碰撞过滤器
        )

        if kitchen_actor is not None and scale is not None:
            try:
                if hasattr(scale, 'tolist'):
                    scale_list = scale.tolist()
                else:
                    scale_list = scale

                if isinstance(scale_list, list) and len(scale_list) == 3:
                    self.gym.set_actor_scale(env_handle, kitchen_actor, float(scale_list[0]))
                elif isinstance(scale_list, (float, int)) or (isinstance(scale_list, list) and len(scale_list) == 1):
                    scale_value = float(scale_list[0] if isinstance(scale_list, list) else scale_list)
                    self.gym.set_actor_scale(env_handle, kitchen_actor, scale_value)
                else:
                    print(f"⚠️ 无法应用缩放 {scale} 到 {urdf_path}，格式不支持")
            except Exception as e:
                print(f"⚠️ 应用缩放时出错: {e}")

        if kitchen_actor is not None:
            successful_actors_list.append(kitchen_actor)
            return True
        return False


    def _init_foot(self):
        """初始化机器人脚部状态"""
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def update_feet_state(self):
        """更新脚部状态"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 确保rigid_body_states_view已正确初始化
        if not hasattr(self, 'rigid_body_states_view') or self.rigid_body_states_view.shape[0] != self.num_envs:
            self.rigid_body_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
            self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)

        # 确保feet_indices有效
        if hasattr(self, 'feet_indices') and len(self.feet_indices) > 0:
            self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
            self.feet_pos = self.feet_state[:, :, :3]
            self.feet_vel = self.feet_state[:, :, 7:10]
        else:
            print("⚠️ 警告: feet_indices未定义或为空")


    def _init_buffers(self):
        # 导入必要的函数
        from legged_gym.utils.isaacgym_utils import get_euler_xyz
        from isaacgym.torch_utils import get_axis_params, to_torch, quat_rotate_inverse

        # 获取gym状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 创建基本包装张量 - 保持原始形状，不重塑
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # 打印张量形状信息用于调试
        print(f"📊 root_states形状: {self.root_states.shape}")
        print(f"📊 dof_state形状: {self.dof_state.shape}")
        print(f"📊 contact_forces形状: {self.contact_forces.shape}")
        print(f"📊 dof_state总元素数: {self.dof_state.numel()}")

        # 为机器人创建视图，而不是修改原始张量
        # 计算每个环境的actor数量
        actors_per_env = self.root_states.shape[0] // self.num_envs

        # 正确区分机器人DOF和总DOF
        # 注意：dof_state可能有不同于预期的结构，需要小心处理
        dofs_per_env = self.dof_state.shape[0] // self.num_envs
        robot_dofs_per_env = self.num_dof  # 机器人的DOF数量

        print(
            f"📊 每个环境的actors: {actors_per_env}, 每个环境的总DOF: {dofs_per_env // 2}, 机器人DOF: {robot_dofs_per_env}")

        # 创建机器人root states视图 - 假设每个环境的第一个actor是机器人
        robot_indices = torch.arange(0, self.root_states.shape[0], actors_per_env, device=self.device)
        self.robot_root_states = self.root_states[robot_indices]
        print(f"📊 机器人root_states形状: {self.robot_root_states.shape}")

        # 从机器人root states创建有用的视图
        self.base_quat = self.robot_root_states[:, 3:7]
        self.base_pos = self.robot_root_states[:, 0:3]

        # 创建DOF状态索引映射 - 关键修改点
        # 为每个环境找出机器人DOF的索引
        self.robot_dof_indices = []
        for env_idx in range(self.num_envs):
            # 假设机器人DOF是每个环境中的前robot_dofs_per_env个DOF
            start_idx = env_idx * dofs_per_env
            self.robot_dof_indices.extend([start_idx + i for i in range(robot_dofs_per_env)])

        # 转换为张量以便更高效的索引
        self.robot_dof_indices = torch.tensor(self.robot_dof_indices, dtype=torch.long, device=self.device)

        # 设置批次大小 - 这里我们设置为1，确保一致性
        self.num_batches = 1
        # self.num_batches = getattr(self.cfg.env, 'num_batches', 1)

        # 尝试创建DOF状态视图 - 使用try-except以适应不同的结构
        try:
            # 尝试创建以环境为单位的视图
            # 注意：根据错误信息，我们怀疑DOF状态可能每个DOF有4个值(而不是2个)，或者有其他不同的结构
            total_states_per_env = self.dof_state.shape[0] // self.num_envs

            # 尝试创建环境视图 - 每个环境有total_states_per_env个状态
            self.dof_state_env_view = self.dof_state.view(self.num_envs, total_states_per_env)

            # 提取机器人DOF状态 - 假设每个环境中前robot_dofs_per_env*2个状态是机器人的
            # 创建机器人专用缓冲区
            # 针对DOF状态处理，不使用reshape，而是直接索引
            self.dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            self.dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)

            # 手动更新DOF状态
            self.manual_dof_update = True

            # 尝试提取机器人DOF状态
            # 可能的结构1：[pos1, vel1, pos2, vel2, ...]
            for env_idx in range(self.num_envs):
                for dof_idx in range(robot_dofs_per_env):
                    pos_idx = dof_idx * 2  # 位置索引
                    vel_idx = dof_idx * 2 + 1  # 速度索引

                    if pos_idx < total_states_per_env and vel_idx < total_states_per_env:
                        self.dof_pos[env_idx, dof_idx] = self.dof_state_env_view[env_idx, pos_idx]
                        self.dof_vel[env_idx, dof_idx] = self.dof_state_env_view[env_idx, vel_idx]

            # 创建视图以在step中使用 - 这只是一个形式上的设置，实际上我们将使用上面的索引
            self.dof_pos_view = self.dof_pos
            self.dof_vel_view = self.dof_vel

            print("✅ 成功创建DOF状态视图和缓冲区")
        except Exception as e:
            print(f"⚠️ 创建DOF状态视图时出错: {e}")
            print("使用直接索引代替视图...")

            # 如果视图创建失败，创建独立的缓冲区并记录需要手动更新
            self.dof_pos = torch.zeros(self.num_envs, robot_dofs_per_env, device=self.device)
            self.dof_vel = torch.zeros(self.num_envs, robot_dofs_per_env, device=self.device)

            # 为step函数设置标记
            self.manual_dof_update = True

            # 创建空视图变量
            self.dof_pos_view = self.dof_pos  # 这只是形式上的赋值
            self.dof_vel_view = self.dof_vel  # 这只是形式上的赋值

        # 清晰地定义训练和推理所需的动作空间大小
        self.num_actions = robot_dofs_per_env

        # 其余初始化
        self.rpy = get_euler_xyz(self.base_quat)
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        dofs_per_env = self.dof_state.shape[0] // self.num_envs
        self.torques = torch.zeros(self.num_envs, dofs_per_env, dtype=torch.float, device=self.device)

        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.robot_root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 关节位置偏移和PD增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.torques = self._compute_torques(self.actions).view(self.torques.shape)

        # 初始化脚部状态
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

        # 创建用于torques的辅助索引
        self.create_torque_indices()

        # 打印训练相关尺寸信息，用于调试
        print(f"🔧 训练动作空间大小: {self.num_actions}")
        print(f"🔧 机器人DOF数量: {self.num_dof}")
        print(f"🔧 动作张量形状: {self.actions.shape}")
        print(f"🔧 每批次样本数: {self.num_batches}")

    def create_torque_indices(self):
        """创建用于扭矩计算的索引映射"""
        # 计算DOF状态张量中每个环境的大小
        states_per_env = self.dof_state.shape[0] // self.num_envs

        # 创建用于扭矩计算的索引映射
        self.torque_indices = []

        # 假设结构1：交替的位置和速度
        for env_idx in range(self.num_envs):
            for dof_idx in range(self.num_actions):
                # 结构1：位置和速度交替
                pos_idx = env_idx * states_per_env + dof_idx * 2
                self.torque_indices.append(pos_idx)

        # 转换为张量
        self.torque_indices = torch.tensor(self.torque_indices, dtype=torch.long, device=self.device)

        # 打印调试信息
        print(f"📊 为扭矩计算创建了 {len(self.torque_indices)} 个索引")
        if len(self.torque_indices) > 0:
            print(f"📊 第一个索引: {self.torque_indices[0]}, 最后一个索引: {self.torque_indices[-1]}")


    def _reset_dofs(self, env_ids):
        """ 重置指定环境的DOF位置和速度 """
        # 设置随机的DOF位置
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        # 获取完整的DOF状态视图
        full_dof_state = self.dof_state.view(self.num_envs, -1, 2)

        # 只更新机器人的DOF
        full_dof_state[env_ids, :self.num_dof, 0] = self.dof_pos[env_ids]
        full_dof_state[env_ids, :self.num_dof, 1] = self.dof_vel[env_ids]

        # 将索引转换为int32
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # 设置DOF状态
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def step(self, actions):
        """
        执行动作、模拟环境，并返回观测、奖励等。
        对于提前终止但尚未达到 rollout_horizon 的环境，
        返回上一次有效的观测，保证每个 rollout 的步数一致。
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        if actions.dim() > 2:
            actions = actions.reshape(-1, self.num_actions)
            if actions.shape[0] != self.num_envs:
                actions = actions[:self.num_envs]

        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_pos[:] = self.dof_pos_view[:]
            self.dof_vel[:] = self.dof_vel_view[:]

        self.post_physics_step()

        # 如果某环境在本 step 已经触发终止（reset_buf 原本置 1），
        # 但 rollout_horizon 尚未到达，则用上一次保存的 last_obs 替换当前 obs
        terminated = (self.reset_buf == 1)
        if terminated.any():
            self.obs_buf[terminated] = self.last_obs[terminated]
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf[terminated] = self.last_privileged_obs[terminated]
            self.rew_buf[terminated] = 0
            # 同时清除 reset_buf，防止下游混淆
            self.reset_buf[terminated] = 0

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        if self.obs_buf.dim() > 2:
            self.obs_buf = self.obs_buf.reshape(self.num_envs, -1)
        if self.privileged_obs_buf is not None and self.privileged_obs_buf.dim() > 2:
            self.privileged_obs_buf = self.privileged_obs_buf.reshape(self.num_envs, -1)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def post_physics_step(self):
        """
        更新状态、计算奖励与终止，但延迟 reset 操作，确保每个 rollout 的长度一致。
        """
        # 刷新状态张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 更新步数计数器
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 提取每个环境中机器人的状态（假设每个环境第一个 actor 是机器人）
        actors_per_env = self.root_states.shape[0] // self.num_envs
        robot_indices = torch.arange(0, self.root_states.shape[0], actors_per_env, device=self.device)
        self.robot_root_states = self.root_states[robot_indices]

        # 更新机器人状态，只针对机器人而非全部 actor
        self.base_pos[:] = self.robot_root_states[:, 0:3]
        self.base_quat[:] = self.robot_root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 回调执行，比如采样指令等
        self._post_physics_step_callback()

        # 检查终止条件（基于接触力、倾斜角度、时间超限等）
        self.check_termination()
        # 计算奖励
        self.compute_reward()

        # 保存当前观测作为最后有效观测（用于后续那些提前终止的环境）
        # 假设 compute_observations() 已经更新了 self.obs_buf 和 self.privileged_obs_buf
        self.last_obs = self.obs_buf.clone()
        if self.privileged_obs_buf is not None:
            self.last_privileged_obs = self.privileged_obs_buf.clone()

        # 延迟 reset：只有当所有环境均达到 rollout_horizon（比如 24 步）时，才调用 reset_idx
        if torch.min(self.episode_length_buf) >= self.cfg.env.rollout_horizon:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.reset_idx(env_ids)
            self.episode_length_buf.fill_(0)
        else:
            # 对于提前触发终止的环境，不立即 reset，而是：
            terminated = (self.reset_buf == 1)
            if terminated.any():
                # 将奖励置 0（或终止奖励），并清除 reset 标记，确保后续 step 返回 last_obs
                self.rew_buf[terminated] = 0
                self.reset_buf[terminated] = 0

        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        # 计算最终观测：对于那些提前终止的环境，仍然返回上一次有效的观测
        self.compute_observations()

        # 保存本次 step 的状态供下次使用
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.robot_root_states[:, 7:13]



    def _get_noise_scale_vec(self, cfg):
        """设置用于缩放噪声的向量"""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        # 设置噪声比例（与观察空间结构匹配）
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[
        9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = 0.  # sin/cos phase

        return noise_vec

    def compute_observations(self):
        """计算观察"""
        # 确保add_noise被初始化
        if not hasattr(self, 'add_noise'):
            self.add_noise = self.cfg.noise.add_noise if hasattr(self.cfg.noise, 'add_noise') else False

        # 如果还没有初始化noise_scale_vec
        if not hasattr(self, 'noise_scale_vec') or self.noise_scale_vec is None:
            self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # 计算sin和cos的相位
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        # 构建观察向量 - 确保只有2D [num_envs, obs_dim]
        self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions,
                                  sin_phase,
                                  cos_phase
                                  ), dim=-1)

        # 构建特权观察向量 - 确保只有2D
        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                             self.base_ang_vel * self.obs_scales.ang_vel,
                                             self.projected_gravity,
                                             self.commands[:, :3] * self.commands_scale,
                                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                             self.dof_vel * self.obs_scales.dof_vel,
                                             self.actions,
                                             sin_phase,
                                             cos_phase
                                             ), dim=-1)

        # 如果需要，添加噪声
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # 确保观察是2D张量 [num_envs, obs_dim]
        if self.obs_buf.dim() > 2:
            print(f"警告: 观察形状为 {self.obs_buf.shape}, 压缩为2D")
            self.obs_buf = self.obs_buf.reshape(self.num_envs, -1)

        if self.privileged_obs_buf is not None and self.privileged_obs_buf.dim() > 2:
            self.privileged_obs_buf = self.privileged_obs_buf.reshape(self.num_envs, -1)

    def _push_robots(self):
        """随机推动机器人。通过设置随机化的基础速度来模拟冲量。"""
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return

        # 计算每个环境的actor数量
        actors_per_env = self.root_states.shape[0] // self.num_envs

        # 最大推动速度
        max_vel = self.cfg.domain_rand.max_push_vel_xy

        # 为推动的环境生成随机速度
        random_vel = torch_rand_float(-max_vel, max_vel, (len(push_env_ids), 2), device=self.device)

        # 计算机器人在root_states中的索引
        robot_indices = torch.arange(0, self.root_states.shape[0], actors_per_env, device=self.device)
        push_robot_indices = robot_indices[push_env_ids]

        # 设置机器人的线性速度 (x/y)
        self.root_states[push_robot_indices, 7:9] = random_vel

        # 转换为int32用于索引
        push_robot_indices_int32 = push_robot_indices.to(dtype=torch.int32)

        # 更新root state
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(push_robot_indices_int32),
                                                     len(push_robot_indices_int32))

    def _post_physics_step_callback(self):
        """在计算终止条件、奖励和观察之前调用的回调"""
        # 首先调用父类方法（如果有）
        super()._post_physics_step_callback()

        # 更新脚部状态
        self.update_feet_state()

        # 计算腿部相位 - 这是我们缺少的属性
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        # 其他必要的计算...

        # 环境ids
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)


    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            # 使用 self.dof_pos 和 self.dof_vel（机器人的状态），self.default_dof_pos 的 shape 为 [1, num_dof]
            robot_torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) \
                            - self.d_gains * self.dof_vel
        elif control_type == "V":
            robot_torques = self.p_gains * (actions_scaled - self.dof_vel) \
                            - self.d_gains * ((self.dof_vel - self.last_dof_vel) / self.sim_params.dt)
        elif control_type == "T":
            robot_torques = actions_scaled
        else:
            raise NameError(f"未知控制器类型: {control_type}")

        robot_torques = torch.clip(robot_torques, -self.torque_limits, self.torque_limits)

        # 创建一个一维张量，其元素数与全局 DOF 数（第一维）一致，即 960
        full_torques = torch.zeros(self.dof_state.shape[0], device=self.device)

        # 将 robot_torques 展平成1D向量
        flat_robot_torques = robot_torques.reshape(-1)

        # 将计算得到的机器人扭矩填入全局扭矩张量中，索引由 self.robot_dof_indices 给出
        full_torques.index_copy_(0, self.robot_dof_indices, flat_robot_torques)

        return full_torques

    def _reward_base_height(self):
        """基础高度奖励"""
        # 确保只使用机器人的高度
        base_height = self.robot_root_states[:, 2]  # 使用机器人的root_states
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_contact(self):
        """接触奖励"""
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        """脚部摆动高度奖励"""
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        """存活奖励"""
        return torch.ones(self.num_envs, device=self.device)

    def _reward_contact_no_vel(self):
        """无速度接触惩罚"""
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    # def _reward_hip_pos(self):
    #     """髋关节位置奖励"""
    #     return torch.sum(torch.square(self.dof_pos[:, [1, 2, 7, 8]]), dim=1)

    def _reward_hip_pos(self):
        """髋关节位置奖励"""
        idxs = [self.dof_names.index(n) for n in [
            'left_hip_roll_joint', 'left_hip_pitch_joint',
            'right_hip_roll_joint', 'right_hip_pitch_joint'
        ]]
        return torch.sum(torch.square(self.dof_pos[:, idxs]), dim=1)

    def compute_reward(self):
        """计算奖励，处理形状不匹配问题"""
        # 初始化奖励为零
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        # 调用每个非零比例的奖励函数
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            try:
                rew = self.reward_functions[i]() * self.reward_scales[name]

                # 确保奖励张量形状正确
                if rew.shape[0] != self.num_envs:
                    print(f"⚠️ 奖励'{name}'的形状({rew.shape})与环境数量({self.num_envs})不匹配，尝试重塑...")
                    # 如果奖励与root_states的数量匹配，则需要提取机器人对应的奖励
                    if rew.shape[0] == self.root_states.shape[0]:
                        actors_per_env = self.root_states.shape[0] // self.num_envs
                        robot_indices = torch.arange(0, self.root_states.shape[0], actors_per_env, device=self.device)
                        rew = rew[robot_indices]

                self.rew_buf += rew
                self.episode_sums[name] += rew

            except Exception as e:
                print(f"❌ 计算奖励'{name}'时出错: {e}")
                # 出错时使用零奖励
                continue

        # 如果只需要正奖励
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        # 添加终止奖励
        if "termination" in self.reward_scales:
            try:
                rew = self._reward_termination() * self.reward_scales["termination"]
                self.rew_buf += rew
                self.episode_sums["termination"] += rew
            except Exception as e:
                print(f"❌ 计算终止奖励时出错: {e}")
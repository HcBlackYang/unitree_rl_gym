import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def train(args):
    # 设置特定于厨房环境的参数
    args.task = "g1_kitchen"
    args.num_envs = 2  # 可以根据GPU显存调整
    args.seed = 1
    args.max_iterations = 1000
    args.headless = False

    # 创建环境和PPO训练器
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # 打印环境信息，使用可靠的方式
    print(f"环境创建完成: num_envs={env.num_envs}, num_actions={env.num_actions}")

    # 使用obs_buf形状来确定观察空间维度
    if hasattr(env, 'obs_buf'):
        obs_dim = env.obs_buf.shape[-1]
        print(f"观察空间维度(从obs_buf): {obs_dim}")
    else:
        print("无法确定观察空间维度: env.obs_buf不存在")

    # 检查其他可能的属性名
    for attr_name in ['observation_dim', 'obs_dim', 'num_obs']:
        if hasattr(env, attr_name):
            print(f"找到观察空间维度: {getattr(env, attr_name)} (属性名: {attr_name})")

    print(f"动作空间维度: {env.num_actions}")

    # 额外检查，确保环境动作空间与期望的机器人DOF一致
    if env.num_actions != env.num_dof:
        print(f"警告: 动作空间大小 ({env.num_actions}) 与机器人DOF数量 ({env.num_dof}) 不匹配!")
        print("这可能会导致训练过程中的形状不匹配错误。")
        # 修复不匹配问题
        env.num_actions = env.num_dof
        print(f"已修正: 动作空间大小设为 {env.num_actions}")

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # 调整训练参数
    train_cfg.runner.save_interval = 50
    train_cfg.runner.eval_interval = 50
    train_cfg.runner.experiment_name = 'g1_kitchen'
    train_cfg.runner.run_name = f'kitchen_nav_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    # 打印PPO配置信息
    print(f"PPO配置: batch_size={train_cfg.algorithm.num_mini_batches}")

    # 检查PPO网络信息
    if hasattr(ppo_runner, 'alg') and hasattr(ppo_runner.alg, 'actor_critic'):
        actor_critic = ppo_runner.alg.actor_critic
        if hasattr(actor_critic, 'num_actions'):
            print(f"PPO策略网络动作维度: {actor_critic.num_actions}")
        elif hasattr(actor_critic, 'action_mean') and hasattr(actor_critic.action_mean, 'out_features'):
            print(f"PPO策略网络动作维度: {actor_critic.action_mean.out_features}")

    # 开始训练
    try:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    except RuntimeError as e:
        if "tensor a" in str(e) and "tensor b" in str(e):
            print("错误: 张量形状不匹配。这可能是由于环境的动作空间与策略网络的输出不匹配导致的。")
            print(f"环境动作维度: {env.num_actions}")

            # 尝试获取环境观察维度
            obs_dim = None
            if hasattr(env, 'obs_buf'):
                obs_dim = env.obs_buf.shape[-1]
            print(f"环境观察维度: {obs_dim}")

            print(f"请检查环境的num_actions和PPO模型的动作输出维度是否一致。")
        raise


if __name__ == '__main__':
    args = get_args()
    train(args)
#!/bin/bash
# unitree_rl_gym/run_kitchen_training.sh

# 设置环境变量
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/blake/Documents/IsaacGym_Preview_4_Package/isaacgym/python/isaacgym/_bindings/linux-x86_64

# 激活环境（如果需要）
#conda activate py38t1110cu113

# 运行训练脚本
python legged_gym/scripts/train_kitchen.py
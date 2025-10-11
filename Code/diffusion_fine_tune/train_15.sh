#!/bin/bash
# train_15.sh
# Stable Diffusion + GAN 训练启动 Shell 脚本
# 功能：后台运行训练、保存日志、查看进程

# -------------------------- 配置参数（可根据需求修改） --------------------------
# 训练启动脚本路径（指向上面写的 run_LoRAGANtrainer.py）
TRAIN_SCRIPT_PATH="/home/jiangxinhai/GMABDA/Code/diffusion_fine_tune/run_LoRAGANtrainer.py"

# 训练参数（可根据需要添加/修改，对应 run_train.py 的命令行参数）
TRAIN_ARGS="
--batch_size 4
--epochs 50
--device cuda:0
"

# -------------------------- 核心逻辑 --------------------------
# 打印启动信息
echo "LoRA-GAN 训练启动脚本"
echo "启动时间：$(date "+%Y-%m-%d_%H:%M:%S")"
echo "启动脚本：${TRAIN_SCRIPT_PATH}"
echo "训练参数：${TRAIN_ARGS}"

# 直接启动训练（前台运行）
echo "启动训练...（tmux中按 Ctrl+B+D 可分离后台）"
python "$TRAIN_SCRIPT_PATH" $TRAIN_ARGS
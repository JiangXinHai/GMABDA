import os
import argparse
import torch
from lora_trainer import LoRATrainer
from Code.common.Config.Configs import diffusionModel_config, run_config, path_config
from Code.common.Utils import logger


def check_environment(args):
    """训练前环境检查"""
    # 1. 检查 GPU
    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，但指定了 GPU 设备，请检查显卡驱动或 PyTorch 安装！")

    # 2. 检查数据集路径
    if not os.path.exists(path_config.DATA_PATHS_IMG_15and17[run_config.INPUT_IMG]):
        raise FileNotFoundError(f"图像路径不存在: {path_config.DATA_PATHS_IMG_15and17[run_config.INPUT_IMG]}")

    # 3. 检查输出目录
    os.makedirs(diffusionModel_config.LORA_SAVE_PATH, exist_ok=True)
    logger.info(f"输出目录准备就绪: {diffusionModel_config.LORA_SAVE_PATH}")


if __name__ == "__main__":
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="LoRA 微调 Stable Diffusion (Twitter15 数据集)")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备 (cuda/cpu/cuda:1)")
    parser.add_argument("--batch_size", type=int, default=None, help=f"批次大小（默认使用 Config 中的 {diffusionModel_config.BATCH_SIZE}）")
    parser.add_argument("--epochs", type=int, default=None, help=f"训练轮次（默认使用 Config 中的 {diffusionModel_config.EPOCHS}）")
    parser.add_argument("--output_dir", type=str, default=None, help=f"LoRA 权重保存目录（默认：{diffusionModel_config.LORA_SAVE_PATH}）")
    args = parser.parse_args()

    # 2. 覆盖配置（如果命令行传入了新值）
    if args.batch_size:
        diffusionModel_config.BATCH_SIZE = args.batch_size
    if args.epochs:
        diffusionModel_config.EPOCHS = args.epochs
    if args.output_dir:
        diffusionModel_config.LORA_SAVE_PATH = args.output_dir

    # 3. 环境检查
    check_environment(args)

    # 4. 初始化 LoRA 训练器
    trainer = LoRATrainer(device=args.device)

    # 5. 开始训练
    trainer.train()
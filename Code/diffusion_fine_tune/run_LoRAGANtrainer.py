# run_train.py
"""
Stable Diffusion + GAN 训练启动脚本
功能：初始化训练器并启动训练，统一管理启动参数
"""
import argparse
import torch

# 导入训练器和日志工具
from Code.diffusion_fine_tune.LoRAGANtrainer import LoRAGANTrainer
from Code.common.Utils import logger


def parse_args():
    """解析命令行参数（支持通过命令行调整关键配置，无需改代码）"""
    parser = argparse.ArgumentParser(description="LoRA-GAN 训练启动脚本")
    
    # 可选参数：可通过命令行覆盖配置文件中的值
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None, 
        help="从指定检查点恢复训练，例：--resume ./save_checkpoint/best_checkpoint.pt"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None, 
        help="临时调整批次大小，优先级高于配置文件"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None, 
        help="临时调整训练轮次，优先级高于配置文件"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="指定训练设备，例：--device cuda:0 或 --device cpu"
    )
    
    return parser.parse_args()


def modify_config(args):
    """根据命令行参数修改配置（覆盖配置文件中的默认值）"""
    from Code.common.Config.Configs import diffusionModel_config as config
    
    # 1. 恢复训练路径（覆盖配置文件的 RESUME_FROM_CHECKPOINT）
    if args.resume is not None:
        config.RESUME_FROM_CHECKPOINT = args.resume
        logger.info(f"命令行指定：从检查点恢复训练 -> {args.resume}")
    
    # 2. 批次大小（覆盖配置文件的 BATCH_SIZE）
    if args.batch_size is not None and args.batch_size > 0:
        config.BATCH_SIZE = args.batch_size
        logger.info(f"命令行指定：批次大小 -> {args.batch_size}")
    
    # 3. 训练轮次（覆盖配置文件的 EPOCHS）
    if args.epochs is not None and args.epochs > 0:
        config.EPOCHS = args.epochs
        logger.info(f"命令行指定：训练轮次 -> {args.epochs}")
    
    # 4. 训练设备（覆盖配置文件的 DEVICE）
    if args.device is not None and args.device in ["cpu", "cuda", "cuda:0", "cuda:1"]:
        # 检查设备是否可用
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"指定设备 {args.device} 不可用，自动切换为 cpu")
            config.DEVICE = "cpu"
        else:
            config.DEVICE = args.device
        logger.info(f"命令行指定：训练设备 -> {config.DEVICE}")
    
    return config

if __name__ == "__main__":
    """主启动逻辑"""
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. 根据参数修改配置
    config = modify_config(args)
    
    # 3. 初始化训练器并启动训练
    logger.info("="*60)
    logger.info("开始启动 LoRA-GAN 训练")
    logger.info(f"当前配置：设备={config.DEVICE} | 批次={config.BATCH_SIZE} | 轮次={config.EPOCHS}")
    logger.info(f"保存路径：{config.LORA_SAVE_PATH}")
    logger.info("="*60)
    
    try:
        trainer = LoRAGANTrainer()
        trainer.start_training()
    except Exception as e:
        logger.error(f"训练异常终止：{str(e)}", exc_info=True)
        raise e
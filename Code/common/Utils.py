import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import re
import glob
import re
import shutil
import torch

class SentimentMapping:
    """情感标签映射工具类（静态方法版），无需实例化即可使用"""
    
    # 静态映射关系（类级别的常量）
    _polarity_mapping = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    _polarity_mapping_back = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    
    @staticmethod
    def num_to_label(num: int) -> str:
        """将数字标签转换为文本标签（静态方法）"""
        if num not in SentimentMapping._polarity_mapping:
            raise ValueError(f"无效的数字标签: {num}，有效标签为 0, 1, 2")
        return SentimentMapping._polarity_mapping[num]
    
    @staticmethod
    def label_to_num(label: str) -> int:
        """将文本标签转换为数字标签（静态方法）"""
        label_lower = label.lower()
        if label_lower not in SentimentMapping._polarity_mapping_back:
            raise ValueError(f"无效的文本标签: {label}，有效标签为 negative, neutral, positive")
        return SentimentMapping._polarity_mapping_back[label_lower]

def add_spaces_around_special_tokens(text):
    """
    为标点符号左右添加空格
    规则：
    1. 常见标点（, . : ; ! ? @ # $ % & * ( )）左右添加空格（避免与单词粘连）
    2. 处理重复空格，最终只保留单个空格
    """
    # 1. 特殊处理$T$后面的情况，确保后面有空格
    text = re.sub(r'(\$T\$)(\S)', r'\1 \2', text)
    
    # 2. 处理标点符号：为标点左右添加空格
    # 定义需要处理的标点
    punc_to_process = r',\. :;! \?@#%& \*\(\)\[\]\{\}<>'
    
    # 正则：匹配标点，确保左右添加空格
    # 使用更严格的模式，确保句首标点前也会添加空格
    text = re.sub(r'(?<!\s)([' + punc_to_process + r'])', r' \1', text)  # 标点前无空格则添加空格
    text = re.sub(r'([' + punc_to_process + r'])(?!\s)', r'\1 ', text)  # 标点后无空格则添加空格
    
    # 3. 清理多余空格：多个空格合并为单个，移除首尾空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def setup_logger():
    # 日志保存目录（自动创建不存在的目录，适配服务器环境）
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../Logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)  # exist_ok=True 避免多进程创建目录冲突
    
    # 日志文件路径（按日期命名，格式：项目名_年-月-日.log）
    current_date = datetime.now().strftime('%Y-%m-%d')  # 获取当前日期（如2025-09-06）
    log_file = os.path.join(log_dir, f'GMABDA_{current_date}.log')
    
    # 配置日志器（避免与其他模块日志冲突，使用项目专属logger名称）
    logger = logging.getLogger('gmabda_logger')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 禁止日志向上传递，避免重复输出
    
    # 检查是否已存在处理器，防止重复添加（适配多次导入场景）
    if not logger.handlers:
        # 文件处理器：按大小切割日志，避免单个文件过大
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 单个日志文件最大10MB（单日超过会自动切割）
            backupCount=10,  # 每个日期的日志最多保留10个切割文件
            encoding='utf-8'  # 支持中文日志，避免乱码
        )
        file_handler.setLevel(logging.INFO)
        
        # 日志格式：包含时间、日志级别、模块名、行号、具体信息（便于定位问题）
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式：年-月-日 时:分:秒
        )
        file_handler.setFormatter(file_formatter)
        
        # 仅添加文件处理器（移除控制台处理器）
        logger.addHandler(file_handler)
    
    return logger


# 初始化日志（项目启动时自动执行，全局只需初始化一次）
logger = setup_logger()
    

class CheckpointUtils:
    @staticmethod
    def should_save_step(step, config):
        if config.SAVE_STRATEGY == "steps" or config.SAVE_STRATEGY == "both":
            return (step + 1) % config.SAVE_STEP == 0
        return False

    @staticmethod
    def should_save_epoch(config):
        return config.SAVE_STRATEGY in ["epoch", "both"]

    @staticmethod
    def save_checkpoint(base_path, model, optimizer, scaler, epoch, step, global_step, best_loss, lora_config, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'lora_config': lora_config,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
        }
        torch.save(checkpoint, f"{base_path}_checkpoint.pt")
        logger.info(f"完整检查点已保存: {base_path}_checkpoint.pt")

    @staticmethod
    def save_lora_weights(base_path, model):
        model.save_pretrained(f"{base_path}_lora")
        logger.info(f"LoRA权重已保存: {base_path}_lora")

    @staticmethod
    def load_checkpoint(checkpoint_path, model, optimizer, scaler, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['epoch'], checkpoint['step'], checkpoint.get('global_step', 0), checkpoint.get('best_loss', float('inf'))

    @staticmethod
    def cleanup_old_checkpoints(save_dir, keep_last_n=5):
        checkpoint_files = glob.glob(f"{save_dir}/epoch_*_checkpoint.pt") + \
                           glob.glob(f"{save_dir}/step_*_checkpoint.pt")
        
        checkpoint_files = [f for f in checkpoint_files if "best" not in os.path.basename(f)]

        def extract_number(filename):
            match = re.search(r'(epoch|step)_(\d+)', filename)
            return int(match.group(2)) if match else 0

        checkpoint_files.sort(key=extract_number)

        if len(checkpoint_files) > keep_last_n:
            for old_file in checkpoint_files[:-keep_last_n]:
                try:
                    os.remove(old_file)
                    lora_dir = old_file.replace('_checkpoint.pt', '_lora')
                    if os.path.exists(lora_dir):
                        shutil.rmtree(lora_dir)
                    logger.info(f"已清理旧检查点: {old_file}")
                except Exception as e:
                    logger.warning(f"清理检查点失败 {old_file}: {e}")

    @staticmethod
    def load_lora_weights(model, lora_path):
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA权重文件不存在: {lora_path}")
        from peft import PeftModel
        return PeftModel.from_pretrained(model, lora_path)
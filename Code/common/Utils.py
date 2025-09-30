import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import re

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
    
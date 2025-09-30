import pandas as pd
import os
from typing import Dict
from PIL import Image
from common.Config.Configs import path_config, run_config
from common.Utils import logger


class TwitterDataset:
    def __init__(self):
        """
        初始化Twitter数据集处理器
        """
        # 初始化路径配置
        self.data = {}  # 存储加载的数据集
    

    def load_data_text(self) -> None:
        """加载文本数据集"""
        logger.info(f"------------load_data_{run_config.INPUT_TEXT}------------")
        if run_config.INPUT_TEXT in self.data:
            logger.info(f"{run_config.INPUT_TEXT} 文本数据集已加载")
            return
            
        # 使用从PathConfig获取的路径加载数据
        path = path_config.DATA_PATHS_TEXT_15and17[run_config.INPUT_TEXT]
        self.data[run_config.INPUT_TEXT] = pd.read_csv(path, sep='\t')
        logger.info(f"成功加载 {run_config.INPUT_TEXT} 数据集，共{len(self.data[run_config.INPUT_TEXT])}条记录")
    
    def __len__(self, split: str) -> int:
        """返回指定划分的数据集长度"""
        return len(self.data[split])
    
    def save_images(self, images, id):
        """
        保存生成的图像
        
        参数:
            images: 图像列表
        """
        output_dir = path_config.DATA_PATHS_IMG_15and17[run_config.OUTPUT_IMG]
        os.makedirs(output_dir, exist_ok=True)
        images[0].save(f"{output_dir}/{id}")
        logger.info(f"已保存 {len(images)} 张图像到 {output_dir}")

    def get_images(self, find_img):
        """
        获取指定文件夹中指定名称的图片并返回Image对象
        :param folder_path: 文件夹路径，如 "D:/images"
        :param target_name: 要查找的图片名称（包含扩展名，如 "test.jpg"）
        :return: PIL.Image对象（找到时），None（未找到时）
        """
        # 遍历文件夹中的文件
        folder_path = path_config.DATA_PATHS_IMG_15and17[run_config.INPUT_IMG]
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # 判断是否是文件且名称匹配
            if os.path.isfile(file_path) and file == find_img:
                try:
                    # 打开图片并返回
                    return Image.open(file_path)
                except Exception as e:
                    logger.info(f"打开图片 {file_path} 出错: {e}")
                    return None
        # 未找到对应名称的图片
        logger.info(f"在 {folder_path} 中未找到 {find_img}")
        return None
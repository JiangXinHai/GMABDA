import pandas as pd
from typing import Dict

class TwitterDataset:
    def __init__(self, data_paths: Dict[str, str]):
        """
        初始化Twitter数据集处理器
        
        参数:
            data_paths: 数据集路径字典，格式为
            {
                "dev": "path/to/dev.csv", 
                "train": "path/to/train.csv", 
                "test": "path/to/test.csv"}
            
        方法：
            load_data(split: str) -> None: 加载指定数据集
            __len__(self, split: str = "train") -> int: 返回指定数据集长度
        """
        self.data_paths = data_paths
        self.data = {}
    

    def load_data(self, split: str) -> None:
        """加载指定划分的数据集"""
        print("------------load_data------------")
        if split not in self.data_paths:
            raise ValueError(f"未知的数据集划分: {split}，可用的划分有: {', '.join(self.data_paths.keys())}")
            
        if split in self.data:
            print(f"{split} 数据集已加载")
            return
            
        path = self.data_paths[split]
        self.data[split] = pd.read_csv(path, sep='\t')
        print(f"成功加载 {split} 数据集，共{len(self.data[split])}条记录")
    
    def __len__(self, split: str) -> int:
        """返回指定划分的数据集长度"""
        print("------------__len__------------")
        self.load_data(split)
        return len(self.data[split])
    
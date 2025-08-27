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
    


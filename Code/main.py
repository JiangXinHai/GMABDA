from TwitterDataset import TwitterDataset
from NewTwitterGenerator import NewTwitterGenerator

if __name__ == "__main__":
    # 创建数据集处理器
    dataset = TwitterDataset()

    # 创建并运行数据生成器
    generator = NewTwitterGenerator(
        dataset=dataset,
    )

    # 生成改写的推特
    generator.generate_tweets()    
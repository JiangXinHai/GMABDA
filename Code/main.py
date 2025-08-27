from TwitterDataset import TwitterDataset
from NewTwitterGenerator import NewTwitterGenerator
from common.Config.Configs import RunConfig

if __name__ == "__main__":
    run_config = RunConfig()
    # 创建数据集处理器
    dataset2015 = TwitterDataset(run_config)

    # 创建并运行数据生成器
    generator = NewTwitterGenerator(
        dataset=dataset2015,
    )

    # 生成改写的推特
    generator.generate_tweets()    
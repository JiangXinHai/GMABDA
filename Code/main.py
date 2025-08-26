import os
from TwitterDataset import TwitterDataset
from NewTwitterGenerator import NewTwitterGenerator

if __name__ == "__main__":
    # 配置参数
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    API_KEY = "wWdRvAHpLUMBHmZUoEVo:JFwiXKjNFxEPNkLKhByj"  # 替换为你的大模型 API密钥
    OUTPUT_DIR_TEXT = os.path.join(cur_dir,"../Data/twitter2015")  # 输出目录
    output_path_text=os.path.join(OUTPUT_DIR_TEXT, "generator_text/new_tweets_dev.tsv")
    OUTPUT_DIR_IMG = os.path.join(cur_dir, "../Data/twitter2015_images")
    output_path_img = os.path.join(OUTPUT_DIR_IMG, "generator_img/new_imgs_dev.tsv")
    # print(cur_dir)
    DATA_PATHS = {
        "dev":   os.path.join(cur_dir,"../Data/twitter2015/dev.tsv"),    # 替换为你的验证集路径
        "train": os.path.join(cur_dir,"../Data/twitter2015/train.tsv"),  # 替换为你的训练集路径
        "test":  os.path.join(cur_dir,"../Data/twitter2015/test.tsv")   # 替换为你的测试集路径
    }
    # print(os.getcwd())
    # for key, path in DATA_PATHS.items():
    #     if os.path.exists(path):
    #         print(f"{key} path exists: {path}")
    #     else:
    #         print(f"{key} path does not exist: {path}")


    # 创建数据集处理器
    dataset2015 = TwitterDataset(DATA_PATHS)

    # 创建并运行数据生成器
    generator = NewTwitterGenerator(
        api_key=API_KEY,
        dataset=dataset2015,
        output_dir=OUTPUT_DIR
    )
    
    # 示例：生成改写的推文
    generator.generate_paraphrased_tweets(
        load_data="dev",
        output_path=os.path.join(OUTPUT_DIR, "new_tweets_dev.tsv")
    )    
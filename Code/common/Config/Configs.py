import torch


class PathConfig:
    # 推文路径
    DATA_PATHS_TEXT_15and17 = {
        "dev_15": "/home/jiangxinhai/GMABDA/Data/twitter2015/dev.tsv",                                 # 15验证集
        "test_15": "/home/jiangxinhai/GMABDA/Data/twitter2015/test.tsv",                               # 15测试集
        "train_15": "/home/jiangxinhai/GMABDA/Data/twitter2015/train.tsv",                             # 15训练集
        "add_dev_15": "/home/jiangxinhai/GMABDA/Data/twitter2015/generator_texts/dev_texts",           # 15验证集生成文本数据
        "add_test_15": "/home/jiangxinhai/GMABDA/Data/twitter2015/generator_texts/test_texts",         # ..后续类似...
        "add_train_15": "/home/jiangxinhai/GMABDA/Data/twitter2015/generator_texts/train_texts",
        "dev_17": "/home/jiangxinhai/GMABDA/Data/twitter2017/dev.tsv",
        "test_17": "/home/jiangxinhai/GMABDA/Data/twitter2017/test.tsv",
        "train_17": "/home/jiangxinhai/GMABDA/Data/twitter2017/train.tsv",
        "add_dev_17": "/home/jiangxinhai/GMABDA/Data/twitter2017/generator_texts/dev_texts",
        "add_test_17": "/home/jiangxinhai/GMABDA/Data/twitter2017/generator_texts/test_texts",
        "add_train_17": "/home/jiangxinhai/GMABDA/Data/twitter2017/generator_texts/trian_texts"
    }

    # 图片路径
    DATA_PATHS_IMG_15and17 = {
        "15": "/home/jiangxinhai/GMABDA/Data/twitter2015_images",
        "add_dev_15": "/home/jiangxinhai/GMABDA/Data/twitter2015_images/generator_imgs/dev_imgs",
        "add_test_15": "/home/jiangxinhai/GMABDA/Data/twitter2015_images/generator_imgs/test_imgs",
        "add_train_15": "/home/jiangxinhai/GMABDA/Data/twitter2015_images/generator_imgs/train_imgs",
        "17": "/home/jiangxinhai/GMABDA/Data/twitter2017_images",
        "add_dev_17": "/home/jiangxinhai/GMABDA/Data/twitter2017_images/generator_imgs/dev_imgs",
        "add_test_17": "/home/jiangxinhai/GMABDA/Data/twitter2017_images/generator_imgs/test_imgs",
        "add_train_17": "/home/jiangxinhai/GMABDA/Data/twitter2017_images/generator_imgs/train_imgs",
    }

class LLMConfig:
    API_KEY = "mVPXuxJlUfqtskruCOQp:vrIvSEmMiaZqtyKETBTM"
    API_URL = "https://spark-api-open.xf-yun.com/v1/chat/completions"
    model_name = "Lite"
    max_retries = 3

class DiffusionModelConfig:
    # ----------推理配置-----------#
    MODEL_NAME = "/home/jiangxinhai/GMABDA/Model/local_stable_diffusion_v1_5"
    NEGATIVE_PROMPT = "blurry, low quality, distorted"
    STRENGTH = 0.8
    NUM_INFERENCE_STEPS = 50
    GUIDANCE_SCALE = 10
    NUM_IMAGES = 1
    SEED = 42

    # ---------微调配置--------------#
    # 1. 数据集配置
    IMAGE_SIZE = 512  # 微调时图像统一尺寸（Stable Diffusion推荐512x512）
    DATA_LOADER_WORKERS = 4  

    # 2. LoRA配置
    LORA_R = 16        # LoRA低秩矩阵维度（常用8/16）
    LORA_ALPHA = 32   # 缩放因子（通常=2*R）
    LORA_DROPOUT = 0.05  #  dropout比例
    LORA_TARGET_MODULES = [  # SD v1.5需训练的注意力模块
        "to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"
    ]
    LORA_SAVE_PATH = "/home/jiangxinhai/GMABDA/Code/diffusion_fine_tune/save_checkpoint"  # 微调后LoRA权重保存路径

    # 3. 训练超参
    BATCH_SIZE = 8    # 批次大小（根据显存调整，12GB卡推荐4）
    EPOCHS = 50       # 训练轮次
    LEARNING_RATE = 9e-5     # 生成器学习率（判别器为2倍）
    WEIGHT_DECAY = 1e-2       # 生成器权重衰减
    WEIGHT_DECAY_DISC = 1e-3  # 判别器权重衰减
    LOGGING_STEP = 10      # 每10步打印一次日志
    SAVE_STEP = 100        # 每100步保存一次LoRA权重
    DEVICE = "cuda"
    DTYPE = ""

    # GAN损失权重
    LAMBDA_GAN = 0.2          # GAN损失权重
    LAMBDA_MATCH = 1.0        # 图文匹配损失权重
    LAMBDA_GP = 10            # R1梯度惩罚权重

    # 新增保存相关配置
    SAVE_STRATEGY = "both"  # "steps", "epoch", "both"
    SAVE_TOTAL_LIMIT = 2     # 最多保存的检查点数量
    SAVE_BEST_ONLY = True    # 是否只保存最佳模型
    RESUME_FROM_CHECKPOINT = None  # 从检查点恢复训练，如 "./lora_weights/best_checkpoint.pt"

    # 临时配置
    NUM_TRAIN_TIMESTEPS=1000
    BETA_START=0.0001
    BETA_END=0.02

class RunConfig:
    INPUT_TEXT = "train_15"
    INPUT_IMG = "15"
    OUTPUT_TEXT = "add_train_15"
    OUTPUT_IMG = "add_train_15"

path_config = PathConfig()
llm_config = LLMConfig()
diffusionModel_config = DiffusionModelConfig()
run_config = RunConfig()
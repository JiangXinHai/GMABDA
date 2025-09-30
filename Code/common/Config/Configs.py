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
    """
        negative_prompt: 负面提示词
        strength: 图像修改强度（0-1之间，值越大修改越明显）
        num_inference_steps: 推理步数
        guidance_scale: 文本的引导尺度(1-20，越大影响越强)
        num_images: 生成图像数量
    """
    MODEL_NAME = "Model/local_stable_diffusion_v1_5"
    NEGATIVE_PROMPT = "blurry, low quality, distorted"
    STRENGTH = 0.8
    NUM_INFERENCE_STEPS = 50
    GUIDANCE_SCALE = 10
    NUM_IMAGES = 1
    SEED = 42

class RunConfig:
    INPUT_TEXT = "train_15"
    INPUT_IMG = "15"
    OUTPUT_TEXT = "add_train_15"
    OUTPUT_IMG = "add_train_15"

path_config = PathConfig()
llm_config = LLMConfig()
diffusionModel_config = DiffusionModelConfig()
run_config = RunConfig()
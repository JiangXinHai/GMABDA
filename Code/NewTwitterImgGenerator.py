from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from common.Config.Configs import diffusionModel_config
from common.Utils import logger
from torchvision import transforms  # PyTorch视觉库的预处理工具

class Image2ImageGenerator:
    """
    使用Stable Diffusion模型进行图像编辑的类
    支持基于现有图像和文本提示生成新图像
    """
    
    def __init__(self, device=None):
        """
        初始化图像编辑生成器
        
        参数:
            model_name: 预训练模型名称或路径
            device: 运行设备，默认为自动检测（优先GPU）
        """
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.generator = torch.Generator("cuda").manual_seed(diffusionModel_config.SEED)
        torch.cuda.empty_cache()  # 清空CUDA缓存

        # 加载图生图管道
        self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            diffusionModel_config.MODEL_NAME,
            safety_checker=None
        ).to(self.device)

        # 启用xFormers高效注意力
        try:
            self.img2img_pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xFormers 已成功启用")
        except Exception as e:
            print(f"xFormers 启用失败！错误：{e}")

        # 启用v1.5支持的内存优化
        self.img2img_pipeline.enable_attention_slicing("max")
    
    def generate_from_image_and_text(self, image, prompt):
        """
        通过现有图像和文本提示生成新图像（图像编辑）
        
        参数:
            image: 基础图像（PIL.Image对象或路径）
            prompt: 文本提示词
            
        返回:
            生成的图像列表（PIL.Image对象）
        """

        # 处理输入图像
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # # 调整图像尺寸为模型所需的尺寸
        # width, height = image.size
        # new_width = (width // 8) * 8
        # new_height = (height // 8) * 8
        # image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # logger.info(f"使用图像尺寸：{new_width}x{new_height}（原尺寸：{width}x{height}）")

        # # PIL图像转PyTorch张量
        # # - ToTensor()：将PIL图像（[0,255]）转为张量（[0,1]），并调整维度为(3, H, W)
        # preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # image_tensor = preprocess(image_resized)  # 此时张量shape: (3, H, W)，值范围[-1,1]
        
        # # 转为半精度（减少内存）+ 增加batch维度 + 移到GPU
        # image_tensor = image_tensor.to(torch.float16).unsqueeze(0).to("cuda")
        # # 最终shape: (1, 3, H, W)，符合模型输入要求（batch=1，3通道，GPU张量）
        
        with torch.no_grad():
            images = self.img2img_pipeline(
                prompt = prompt,
                image = image,
                negative_prompt = diffusionModel_config.NEGATIVE_PROMPT,
                strength = diffusionModel_config.STRENGTH,
                num_inference_steps = diffusionModel_config.NUM_INFERENCE_STEPS,
                guidance_scale = diffusionModel_config.GUIDANCE_SCALE,
                num_images_per_prompt = diffusionModel_config.NUM_IMAGES,
                generator = self.generator
            ).images

        return images
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from Code.common.Config.Configs import diffusionModel_config
from Code.common.Utils import logger
from torchvision import transforms  # PyTorch视觉库的预处理工具
from peft import PeftModel

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

        # 确定数据类型
        self.dtype = torch.float32
        
        self.generator = torch.Generator(self.device).manual_seed(diffusionModel_config.SEED)
        torch.cuda.empty_cache()  # 清空CUDA缓存

        # 加载图生图管道
        self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            diffusionModel_config.MODEL_NAME,
            safety_checker=None
        ).to(self.device)

        # 加载LoRA权重
        self.img2img_pipeline.unet = PeftModel.from_pretrained(
            self.img2img_pipeline.unet,
            diffusionModel_config.LORA_SAVE_PATH + '/best_lora',
            torch_dtype=self.dtype
        )

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
        
        # 调用尺寸调整方法
        resized_image = self._resize_image(image)

        with torch.no_grad():
            images = self.img2img_pipeline(
                prompt = prompt,
                image = resized_image,
                negative_prompt = diffusionModel_config.NEGATIVE_PROMPT,
                strength = diffusionModel_config.STRENGTH,
                num_inference_steps = diffusionModel_config.NUM_INFERENCE_STEPS,
                guidance_scale = diffusionModel_config.GUIDANCE_SCALE,
                num_images_per_prompt = diffusionModel_config.NUM_IMAGES,
                generator = self.generator
            ).images

        return images
    
    def _resize_image(self, image):
        """调整图像尺寸：正方形则(512,512)，非正方形短边512"""
        width, height = image.size
        if width == height:
            # 正方形，直接缩放到512x512
            new_size = (512, 512)
        else:
            # 非正方形，短边为512，长边等比例缩放
            if width < height:
                new_width = 512
                new_height = int(height * (512 / width))
            else:
                new_height = 512
                new_width = int(width * (512 / height))
            # 确保尺寸是8的倍数（Stable Diffusion要求）
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            new_size = (new_width, new_height)
        logger.info(f"原图像尺寸：{width}x{height}，调整后尺寸：{new_size[0]}x{new_size[1]}")
        return image.resize(new_size, Image.Resampling.LANCZOS)
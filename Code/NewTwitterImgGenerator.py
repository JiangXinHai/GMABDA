from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
from common.Config.Configs import PathConfig, DiffusionModelConfig, RunConfig
from TwitterDataset import TwitterDataset

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
        self.diffusion_model_config = DiffusionModelConfig()
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 加载图生图管道
        self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.diffusion_model_config.MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # 设置安全检查器（简单版本）
        self.img2img_pipeline.safety_checker = self.safety_checker
    
    def safety_checker(self, images, clip_input):
        """自定义安全检查器（返回原图）"""
        return images, [False] * len(images)
    
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
        
        # 调整图像尺寸为模型所需的尺寸
        image = image.resize((512, 512))
        
        with torch.no_grad():
            images = self.img2img_pipeline(
                prompt = prompt,
                image = image,
                negative_prompt = self.diffusion_model_config.NEGATIVE_PROMPT,
                strength = self.diffusion_model_config.STRENGTH,
                num_inference_steps = self.diffusion_model_config.NUM_INFERENCE_STEPS,
                guidance_scale = self.diffusion_model_config.GUIDANCE_SCALE,
                num_images_per_prompt = self.diffusion_model_config.NUM_IMAGES
            ).images
            
        return images

# 使用示例
if __name__ == "__main__":
    
    # 初始化生成器
    generator = Image2ImageGenerator()
    
    # 通过图像和文本生成新图像
    input_image_path = "/home/jiangxinhai/GMABDA/Data/twitter2015_images/1032623.jpg"  # 替换为你的输入图像路径
    img_prompt = "accompanying image of \"RT @ TPM : Court to censure Montana judge over rape comments\""
    
    edited_images = generator.generate_from_image_and_text(
        image=input_image_path,
        prompt=img_prompt,
    )
    run_config = RunConfig()
    data = TwitterDataset(run_config)
    data.save_images(edited_images, "1.jpg")
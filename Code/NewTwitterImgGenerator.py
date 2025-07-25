from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os

class Image2ImageGenerator:
    """
    使用Stable Diffusion模型进行图像编辑的类
    支持基于现有图像和文本提示生成新图像
    """
    
    def __init__(self, model_name="Model/local_stable_diffusion_v1_5", device=None):
        """
        初始化图像编辑生成器
        
        参数:
            model_name: 预训练模型名称或路径
            device: 运行设备，默认为自动检测（优先GPU）
        """
        self.model_name = model_name
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 加载图生图管道
        self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # 设置安全检查器（简单版本）
        self.img2img_pipeline.safety_checker = self.safety_checker
    
    def safety_checker(self, images, clip_input):
        """自定义安全检查器（返回原图）"""
        return images, [False] * len(images)
    
    def generate_from_image_and_text(self, image, prompt, negative_prompt="", 
                                    strength=0.8, num_inference_steps=250,
                                    guidance_scale=15, num_images=5):
        """
        通过现有图像和文本提示生成新图像（图像编辑）
        
        参数:
            image: 基础图像（PIL.Image对象或路径）
            prompt: 文本提示词
            negative_prompt: 负面提示词
            strength: 图像修改强度（0-1之间，值越大修改越明显）
            num_inference_steps: 推理步数
            guidance_scale: 文本的引导尺度(1-20，越大影响越强)
            num_images: 生成图像数量
            
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
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images
            ).images
            
        return images
    
    def save_images(self, images, output_dir="/home/jiangxinhai/GMABDA/Data/twitter2015_images/generator_img", prefix="edited"):
        """
        保存生成的图像
        
        参数:
            images: 图像列表
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            img.save(f"{output_dir}/{prefix}_{i}.png")
        print(f"已保存 {len(images)} 张图像到 {output_dir}")


# 使用示例
if __name__ == "__main__":
    
    # 初始化生成器
    generator = Image2ImageGenerator()
    
    # 通过图像和文本生成新图像
    input_image_path = "/home/jiangxinhai/GMABDA/Data/twitter2015_images/1032623.jpg"  # 替换为你的输入图像路径
    img_prompt = "accompanying image of \"RT @ TPM : Court to censure Montana judge over rape comments\""
    negative_prompt = "blurry, low quality, distorted, "
    
    edited_images = generator.generate_from_image_and_text(
        image=input_image_path,
        prompt=img_prompt,
        negative_prompt=negative_prompt,
    )
    
    generator.save_images(edited_images, prefix="style_transfer")
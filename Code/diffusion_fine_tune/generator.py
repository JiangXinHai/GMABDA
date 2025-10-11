import torch
import os
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
from peft import LoraConfig, get_peft_model

# 导入你的工具类（日志、随机种子等）
from Code.common.Utils import logger
# 导入配置（含 LORA_TARGET_MODULES 等参数）
from Code.common.Config.Configs import diffusionModel_config


class LoRADiffusionGenerator:
    """
    Stable Diffusion 生成器：仅用 LoRA 微调 UNet 关键层
    核心功能：图像-潜空间转换、文本编码、LoRA 权重管理
    """
    def __init__(self):
        # 基础设备与数据类型配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        logger.info(f"生成器初始化：设备={self.device}，数据类型={self.dtype}")

        # -------------------------- 1. 加载 VAE（冻结）--------------------------
        # VAE 负责图像与潜空间的转换，无需微调
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=diffusionModel_config.MODEL_NAME,
            subfolder="vae",
            torch_dtype=self.dtype
        ).to(self.device)
        # 冻结 VAE 所有参数
        for param in self.vae.parameters():
            param.requires_grad = False
        logger.info("VAE 加载完成（已冻结）")

        # -------------------------- 2. 加载文本编码器与 Tokenizer（冻结）--------------------------
        # 加载完整 Pipeline 以获取文本编码器和 Tokenizer（无需生成功能）
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=diffusionModel_config.MODEL_NAME,
            safety_checker=None,  # 关闭安全检查，提升效率
            torch_dtype=self.dtype,
            use_safetensors=True  # 优先使用 safetensors 权重，更安全
        ).to(self.device)

        # 冻结文本编码器（CLIP Text Encoder）
        for param in self.pipeline.text_encoder.parameters():
            param.requires_grad = False
        logger.info("文本编码器（CLIP）加载完成（已冻结）")

        # -------------------------- 3. 配置 LoRA 并包装 UNet（核心微调部分）--------------------------
        # LoRA 配置：仅针对 UNet 的注意力层和残差投影层
        self.lora_config = LoraConfig(
            r=diffusionModel_config.LORA_R,  # LoRA 低秩矩阵维度
            lora_alpha=diffusionModel_config.LORA_ALPHA,  # 缩放因子（通常为 2*r）
            target_modules=diffusionModel_config.LORA_TARGET_MODULES,  # 目标微调层（你的配置：to_k/to_q/to_v 等）
            lora_dropout=diffusionModel_config.LORA_DROPOUT,  # Dropout 比例，防止过拟合
            bias="none",  # 不微调偏置项
            task_type="IMAGE_2_IMAGE"  # 任务类型，适配 img2img 场景
        )

        # 用 LoRA 包装 UNet，仅微调目标层
        self.unet = get_peft_model(
            model=self.pipeline.unet,
            peft_config=self.lora_config
        ).to(self.device)

        # 打印可训练参数比例（验证 LoRA 生效）
        trainable_params, all_param = self.unet.get_nb_trainable_parameters()
        logger.info(
            f"UNet LoRA 配置完成：可训练参数 {trainable_params / (all_param + 1e-8) * 100:.4f}% "
            f"({trainable_params}/{all_param})"
        )

        # -------------------------- 4. 暴露核心组件（供训练调用）--------------------------
        self.tokenizer = self.pipeline.tokenizer  # 文本 Tokenizer
        self.text_encoder = self.pipeline.text_encoder  # 文本编码器（已冻结）

    def image_to_latent(self, image_tensor):
        """
        将 RGB 图像张量（[0,1]）编码为 Stable Diffusion 潜空间（latent）
        Args:
            image_tensor: 输入图像张量，形状 [batch, 3, H, W]，值范围 [0,1]
        Returns:
            latent_tensor: 潜空间张量，形状 [batch, 4, H/8, W/8]（SD 固定下采样 8 倍）
        """
        # 1. 图像归一化：从 [0,1] 转 [-1,1]（适配 VAE 输入要求）
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)
        normalized_image = (image_tensor * 2.0) - 1.0  # 缩放公式

        # 2. VAE 编码：生成潜空间分布并采样
        with torch.no_grad():  # VAE 已冻结，无需计算梯度
            vae_encoding = self.vae.encode(normalized_image)
            latent_tensor = vae_encoding.latent_dist.sample()  # 从高斯分布采样

        # 3. 潜空间缩放：SD 官方固定缩放因子（0.18215），确保 latent 分布稳定
        latent_tensor = latent_tensor * 0.18215

        return latent_tensor

    def latent_to_image(self, latent_tensor):
        """
        将潜空间张量（latent）解码为 RGB 图像张量（[0,1]）
        Args:
            latent_tensor: 潜空间张量，形状 [batch, 4, H/8, W/8]
        Returns:
            image_tensor: 输出图像张量，形状 [batch, 3, H, W]，值范围 [0,1]
        """
        # 1. 潜空间逆缩放：还原 VAE 输入尺度
        latent_tensor = latent_tensor.to(self.device, dtype=self.dtype)
        scaled_latent = latent_tensor / 0.18215  # 逆操作：除以固定缩放因子

        # 2. VAE 解码：从潜空间生成图像
        with torch.no_grad():
            vae_decoding = self.vae.decode(scaled_latent)
            normalized_image = vae_decoding.sample  # 解码输出，值范围 [-1,1]

        # 3. 图像归一化：从 [-1,1] 转 [0,1]，并裁剪异常值
        image_tensor = (normalized_image + 1.0) / 2.0  # 缩放公式
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)  # 防止数值溢出（0-1 范围）

        return image_tensor

    def get_text_embeddings(self, text_list):
        """
        将文本列表编码为 CLIP 文本嵌入（768 维），用于 UNet 条件输入
        Args:
            text_list: 文本列表，如 ["a photo of a cat", "a dog playing"]
        Returns:
            text_embeds: 文本嵌入张量，形状 [batch, 77, 768]（77 为 CLIP 最大文本长度）
        """
        # 1. 文本 Tokenize：转换为模型可识别的 ID
        text_inputs = self.tokenizer(
            text=text_list,
            padding="max_length",  # 填充到 CLIP 最大长度（77）
            max_length=self.tokenizer.model_max_length,
            truncation=True,  # 截断过长文本
            return_tensors="pt"  # 返回 PyTorch 张量
        ).to(self.device)

        # 2. 文本编码：生成 768 维嵌入
        with torch.no_grad():  # 文本编码器已冻结，无需计算梯度
            text_embeds = self.text_encoder(text_inputs.input_ids)[0]  # [0] 取编码输出（忽略池化输出）

        return text_embeds

    def get_negative_text_embeddings(self, text_list):
        """
        生成负样本文本嵌入（打乱文本-图像对应关系），用于图文匹配损失计算
        Args:
            text_list: 原始文本列表（与图像对应的文本）
        Returns:
            negative_text_embeds: 负样本文本嵌入，形状 [batch, 77, 768]
        """
        # 1. 打乱文本顺序：构建“图像-错误文本”对
        shuffled_indices = torch.randperm(n=len(text_list), device=self.device)  # 随机打乱索引
        shuffled_text_list = [text_list[idx] for idx in shuffled_indices]  # 按打乱索引重排文本

        # 2. 编码负样本文本（复用现有文本编码逻辑）
        negative_text_embeds = self.get_text_embeddings(shuffled_text_list)

        return negative_text_embeds

    def load_lora_weights(self, lora_weight_path):
        """
        加载预训练的 LoRA 权重（用于推理或继续训练）
        Args:
            lora_weight_path: LoRA 权重路径（如 "./save_lora/best_lora"）
        Returns:
            self.unet: 加载权重后的 UNet（带 LoRA）
        """
        from peft import PeftModel  # 延迟导入，避免未使用时加载

        # 检查权重路径是否存在
        if not os.path.exists(lora_weight_path):
            error_msg = f"LoRA 权重路径不存在：{lora_weight_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 加载 LoRA 权重
        self.unet = PeftModel.from_pretrained(
            model=self.unet,
            model_id=lora_weight_path,
            torch_dtype=self.dtype
        ).to(self.device)
        logger.info(f"成功加载 LoRA 权重：{lora_weight_path}")

        return self.unet

    def save_lora_weights(self, save_path):
        """
        保存当前训练的 LoRA 权重（仅保存 LoRA 增量权重，体积小）
        Args:
            save_path: 保存路径（如 "./save_lora/epoch_1_lora"）
        """
        # 创建保存目录（若不存在）
        os.makedirs(save_path, exist_ok=True)

        # 保存 LoRA 权重（仅增量部分，约几 MB 到几十 MB）
        self.unet.save_pretrained(save_path)
        logger.info(f"LoRA 权重已保存至：{save_path}")

    def predict_noise(self, noisy_latents, timesteps, text_embeds):
        """封装 UNet 推理，统一调用方式"""
        return self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds)

# -------------------------- 测试代码（可选，验证生成器功能）--------------------------
# if __name__ == "__main__":
#     # 初始化生成器
#     generator = LoRADiffusionGenerator()

#     # 1. 测试文本编码
#     test_texts = ["a photo of a cat on a chair", "a dog playing in the park"]
#     text_embeds = generator.get_text_embeddings(test_texts)
#     logger.info(f"文本嵌入形状：{text_embeds.shape}（预期：[2, 77, 768]）")

#     # 2. 测试负样本文本编码
#     negative_embeds = generator.get_negative_text_embeddings(test_texts)
#     logger.info(f"负样本文本嵌入形状：{negative_embeds.shape}（预期：[2, 77, 768]）")

#     # 3. 测试图像-潜空间转换（需准备测试图像）
#     try:
#         from torchvision import transforms
#         from PIL import Image

#         # 加载测试图像（替换为你的图像路径）
#         test_image = Image.open("/home/jiangxinhai/GMABDA/Code/diffusion_fine_tune/test.jpg").convert("RGB")
#         # 预处理： resize 到配置尺寸，转张量
#         transform = transforms.Compose([
#             transforms.Resize((diffusionModel_config.IMAGE_SIZE, diffusionModel_config.IMAGE_SIZE)),
#             transforms.ToTensor(),  # 转 [0,1] 张量
#         ])
#         image_tensor = transform(test_image).unsqueeze(0)  # 增加 batch 维度 [1,3,H,W]

#         # 图像转潜空间
#         latent = generator.image_to_latent(image_tensor)
#         logger.info(f"潜空间张量形状：{latent.shape}（预期：[1,4,{diffusionModel_config.IMAGE_SIZE//8},{diffusionModel_config.IMAGE_SIZE//8}]）")

#         # 潜空间转图像
#         recon_image = generator.latent_to_image(latent)
#         logger.info(f"重构图像张量形状：{recon_image.shape}（预期：[1,3,{diffusionModel_config.IMAGE_SIZE},{diffusionModel_config.IMAGE_SIZE}]）")

#         # 保存重构图像（验证转换正确性）
#         recon_pil = transforms.ToPILImage()(recon_image[0])  # 移除 batch 维度
#         recon_pil.save("/home/jiangxinhai/GMABDA/Code/diffusion_fine_tune/recon_test_image.jpg")
#         logger.info("重构图像已保存至：/home/jiangxinhai/GMABDA/Code/diffusion_fine_tune/recon_test_image.jpg")

#     except Exception as e:
#         logger.warning(f"图像转换测试跳过：{str(e)}（需准备测试图像）")

#     logger.info("生成器功能测试完成")
import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import DDPMScheduler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter 
from datetime import datetime

# 导入自定义模块
from Code.TwitterDataset import TwitterDataset  # 你的数据集类
from Code.common.Utils import logger, CheckpointUtils  # 工具类（日志、检查点）
from Code.common.Config.Configs import diffusionModel_config as config  # 配置
from Code.diffusion_fine_tune.generator import LoRADiffusionGenerator  # 生成器
from Code.diffusion_fine_tune.discriminator import StyleGAN2Discriminator  # 判别器


class LoRAGANTrainer:
    """
    扩散模型+GAN混合训练器：
    - 生成器：LoRA微调的Stable Diffusion UNet
    - 判别器：双输出StyleGAN2判别器（真实性+图文匹配）
    - 核心逻辑：交替训练判别器与生成器，支持检查点续训、最佳模型保存
    """
    def __init__(self):
        # -------------------------- 1. 初始化核心组件 --------------------------
        # 生成器（LoRA+SD UNet）
        self.generator = LoRADiffusionGenerator()
        # 判别器（双输出StyleGAN2）
        self.discriminator = StyleGAN2Discriminator().to(self.generator.device)
        # 扩散调度器（DDPMScheduler，与SD匹配）
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config.MODEL_NAME,
            subfolder="scheduler"
        )
        # 关键：确保调度器的prediction_type与生成器一致（默认"epsilon"，与SD匹配）
        self.noise_scheduler.config.prediction_type = "epsilon"

        # -------------------------- 新增：TensorBoard 初始化 --------------------------
        # TensorBoard 日志保存路径
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.tb_log_dir = f"/home/jiangxinhai/GMABDA/Logs/LoRAdiffusionGAN_tensorboard_logs/tb_{timestamp})"
        os.makedirs(self.tb_log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=self.tb_log_dir)
        logger.info(f"TensorBoard 初始化完成，日志路径：{self.tb_log_dir}\n(启动命令：tensorboard --logdir {self.tb_log_dir})")
        # -------------------------- 2. 数据集加载 --------------------------
        self.dataset = TwitterDataset(is_train=True)
        self.train_loader, self.val_loader = self.dataset.build_train_val_dataloaders()
        logger.info(f"数据集加载完成：训练集{len(self.train_loader)}批，验证集{len(self.val_loader)}批")

        # -------------------------- 3. 优化器配置 --------------------------
        # 生成器优化器（仅优化UNet的LoRA层）
        self.optimizer_gen = AdamW(
            params=self.generator.unet.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            eps=1e-8  # 防止梯度爆炸
        )
        # 判别器优化器（全量训练，学习率是生成器的2倍）
        self.optimizer_disc = AdamW(
            params=self.discriminator.parameters(),
            lr=config.LEARNING_RATE * 2,  # GAN常用配置：判别器LR略高
            weight_decay=config.WEIGHT_DECAY_DISC if hasattr(config, "WEIGHT_DECAY_DISC") else 1e-3,
            eps=1e-8
        )

        # -------------------------- 4. 混合精度训练 --------------------------
        self.scaler = torch.cuda.amp.GradScaler() if self.generator.device == "cuda" else None
        if self.scaler:
            logger.info("启用混合精度训练（AMP），加速训练并节省显存")

        # -------------------------- 5. 训练状态初始化 --------------------------
        self.current_epoch = 0  # 当前轮次
        self.global_step = 0     # 全局步数
        self.best_val_loss = float('inf')  # 最佳验证损失（用于保存最佳模型）
        self.device = self.generator.device  # 统一设备

        # -------------------------- 6. 输出目录与检查点恢复 --------------------------
        os.makedirs(config.LORA_SAVE_PATH, exist_ok=True)
        if config.RESUME_FROM_CHECKPOINT and os.path.exists(config.RESUME_FROM_CHECKPOINT):
            self._load_checkpoint(config.RESUME_FROM_CHECKPOINT)
            logger.info(f"从检查点恢复训练：{config.RESUME_FROM_CHECKPOINT}")
        else:
            logger.info("未指定检查点或路径不存在，将从头开始训练")

        logger.info("训练器初始化完成，等待启动训练")

    def _prepare_visualization_data(self, batch, fake_imgs):
        """
        准备可视化数据：
        - 随机挑选 2-4 个样本（避免可视化太占内存）
        - 处理图像格式（从[0,1]张量转成TensorBoard可显示格式）
        - 关联文本标签
        """
        # 1. 随机挑选样本（每次选2个，避免网格过大）
        batch_size = batch["input_image"].shape[0]
        sample_indices = random.sample(range(batch_size), k=min(2, batch_size))  # 选2个样本

        # 2. 提取数据（真实图像、生成图像、文本）
        real_imgs = batch["input_image"][sample_indices]  # [2,3,H,W]，[0,1]
        fake_imgs = fake_imgs[sample_indices]            # [2,3,H,W]，[0,1]
        prompts = [batch["text_prompt"][i] for i in sample_indices]  # 文本列表

        # 3. 图像格式处理（TensorBoard要求[H,W,C]，且值在[0,1]）
        # 转置维度：[C,H,W] → [H,W,C]
        real_imgs = real_imgs.permute(0, 2, 3, 1).cpu()  # 移到CPU，避免显存占用
        fake_imgs = fake_imgs.permute(0, 2, 3, 1).cpu()

        # 4. 组合成网格标题（文本标签）
        grid_titles = [f"Real: {p[:20]}..." for p in prompts] + [f"Fake: {p[:20]}..." for p in prompts]

        # 5. 拼接真实/生成图像（按行拼接：real1, fake1; real2, fake2）
        combined_imgs = []
        for r, f in zip(real_imgs, fake_imgs):
            combined_imgs.extend([r, f])  # 每个样本对应「真实+生成」2张图

        return torch.stack(combined_imgs), grid_titles  # [4, H, W, 3], 标题列表
    
    def _create_image_grid(self, imgs, titles):
        """
        创建图像网格（用matplotlib绘制，支持添加文本标题）
        - imgs: [N, H, W, C] 张量，N为图像数量
        - titles: 列表，每个图像的标题
        """
        import matplotlib.pyplot as plt  # 延迟导入，避免启动时加载
        plt.switch_backend('Agg')  # 非交互式后端，避免弹出窗口

        # 计算网格尺寸（这里固定2列，N行）
        num_imgs = imgs.shape[0]
        num_cols = 2  # 每列2张图（real + fake）
        num_rows = num_imgs // num_cols

        # 创建画布
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6*num_rows))
        axes = axes.flatten()  # 展平axes，方便循环

        # 填充图像和标题
        for idx, (img, title) in enumerate(zip(imgs, titles)):
            axes[idx].imshow(img.numpy())  # 显示图像
            axes[idx].set_title(title, fontsize=8, wrap=True)  # 添加标题（自动换行）
            axes[idx].axis('off')  # 隐藏坐标轴

        # 调整间距
        plt.tight_layout()
        return fig

    def _train_discriminator(self, batch):
        """
        单批数据训练判别器：
        输入：batch（含图像和文本）
        输出：判别器总损失
        """
        # 1. 数据预处理
        input_imgs = batch["input_image"].to(self.device)  # [B,3,H,W]，值范围[0,1]
        prompts = batch["text_prompt"]                     # 文本列表

        # 2. 生成必要特征（调用生成器方法）
        real_latents = self.generator.image_to_latent(input_imgs)  # 真实图像→latent
        text_embeds = self.generator.get_text_embeddings(prompts)  # 正文本嵌入 [B,77,768]
        neg_text_embeds = self.generator.get_negative_text_embeddings(prompts)  # 负文本嵌入

        # 3. 生成假图像（用于判别器训练）
        with torch.no_grad():  # 生成假图像时不计算生成器梯度
            # 扩散过程：加噪声→UNet预测噪声→去噪得到假latent
            noise = torch.randn_like(real_latents)
            timestep_scalar = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                size=(), device=self.device
            ).item()
            timestep_batch = torch.tensor([timestep_scalar] * real_latents.shape[0], device=self.device)

            noisy_latents = self.noise_scheduler.add_noise(real_latents, noise, timestep_batch)
            noise_pred = self.generator.predict_noise(noisy_latents, timestep_batch, text_embeds).sample
            # 关键3：调用step时传入「标量时间步」，获取去噪后的latent
            denoise_out = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep_scalar,  # 传入标量（适配单时间步step）
                sample=noisy_latents,
                return_dict=True  # 显式返回字典，避免元组格式不匹配
            )
            fake_latents = denoise_out.prev_sample  # 从返回字典中取prev_sample

            # latent→图像（用于判别器输入）
            fake_imgs = self.generator.latent_to_image(fake_latents)  # [B,3,H,W]
            real_imgs = self.generator.latent_to_image(real_latents)  # 真实图像（用于对比）

        # 4. 判别器前向传播（计算三类样本分数）
        # 4.1 真实图像+正文本（标签1：真实且匹配）
        real_pred_pos, match_pred_pos = self.discriminator(real_imgs, text_embeds)
        # 4.2 真实图像+负文本（标签0：真实但不匹配）
        _, match_pred_neg = self.discriminator(real_imgs, neg_text_embeds)
        # 4.3 假图像+正文本（标签0：虚假且不匹配）
        fake_pred, match_pred_fake = self.discriminator(fake_imgs.detach(), text_embeds)  # detach()避免生成器梯度

        # 5. 计算判别器损失
        # 5.1 真实性损失（二分类交叉熵）
        loss_real = F.binary_cross_entropy_with_logits(real_pred_pos, torch.ones_like(real_pred_pos))
        loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        loss_gan = (loss_real + loss_fake) / 2  # GAN基础损失

        # 5.2 图文匹配损失（二分类交叉熵）
        loss_match_pos = F.binary_cross_entropy_with_logits(match_pred_pos, torch.ones_like(match_pred_pos))
        loss_match_neg = F.binary_cross_entropy_with_logits(match_pred_neg, torch.zeros_like(match_pred_neg))
        loss_match_fake = F.binary_cross_entropy_with_logits(match_pred_fake, torch.zeros_like(match_pred_fake))
        loss_match = (loss_match_pos + loss_match_neg + loss_match_fake) / 3  # 平均三类匹配损失

        # 5.3 R1梯度惩罚（稳定判别器训练）
        r1_penalty = self.discriminator.compute_r1_penalty(real_imgs, text_embeds)

        # 5.4 总损失（加权求和）
        total_loss = loss_gan + \
                     config.LAMBDA_MATCH * loss_match + \
                     config.LAMBDA_GP * r1_penalty

        # 6. 判别器反向传播与参数更新
        self.optimizer_disc.zero_grad()  # 清空梯度
        if self.scaler:
            self.scaler.scale(total_loss).backward()  # 混合精度：缩放损失
            self.scaler.step(self.optimizer_disc)      # 更新参数
        else:
            total_loss.backward()  # 普通精度：直接反向传播
            self.optimizer_disc.step()
 # -------------------------- 记录判别器损失到TensorBoard --------------------------
        self.tb_writer.add_scalar("Discriminator/Total_Loss", total_loss.item(), self.global_step)
        self.tb_writer.add_scalar("Discriminator/GAN_Loss", loss_gan.item(), self.global_step)
        self.tb_writer.add_scalar("Discriminator/Match_Loss", loss_match.item(), self.global_step)
        self.tb_writer.add_scalar("Discriminator/R1_Penalty", r1_penalty.item(), self.global_step)

        return total_loss.item(), loss_gan.item(), loss_match.item(), r1_penalty.item()

    def _train_generator(self, batch):
        """
        单批数据训练生成器：
        输入：batch（含图像和文本）
        输出：生成器总损失
        """
        # 1. 数据预处理（与判别器一致）
        input_imgs = batch["input_image"].to(self.device)
        prompts = batch["text_prompt"]

        # 2. 生成必要特征
        real_latents = self.generator.image_to_latent(input_imgs)
        text_embeds = self.generator.get_text_embeddings(prompts)

        # 3. 扩散过程（核心：预测噪声）
        noise = torch.randn_like(real_latents)  # 生成随机噪声
        # 生成「单时间步标量」（批量内所有样本用同一时间步）
        timestep_scalar = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            size=(),  # 生成标量
            device=self.device
        ).item()  # 转为Python int
        
        # 生成「同一时间步的批量张量」（用于add_noise）
        timestep_batch = torch.tensor([timestep_scalar] * real_latents.shape[0], device=self.device)
        noisy_latents = self.noise_scheduler.add_noise(real_latents, noise, timestep_batch)  # 加噪声

        # 4. 生成器前向传播（UNet预测噪声 + 生成假图像）
        noise_pred = self.generator.predict_noise(noisy_latents, timestep_batch, text_embeds).sample
        # 调用step时传入「标量时间步」，获取去噪后的latent
        denoise_out = self.noise_scheduler.step(
            model_output=noise_pred,
            timestep=timestep_scalar,  # 传入标量（适配单时间步step）
            sample=noisy_latents,
            return_dict=True  # 显式返回字典，避免元组格式不匹配
        )
        fake_latents = denoise_out.prev_sample  # 从返回字典中取prev_sample
        fake_imgs = self.generator.latent_to_image(fake_latents)

        # 5. 计算生成器损失
        # 5.1 扩散损失（MSE：预测噪声与真实噪声的差距）
        loss_diffusion = F.mse_loss(noise_pred, noise)

        # 5.2 GAN损失（欺骗判别器：让判别器认为假图像是真实的）
        fake_pred_gen, match_pred_gen = self.discriminator(fake_imgs, text_embeds)
        loss_gan = F.binary_cross_entropy_with_logits(fake_pred_gen, torch.ones_like(fake_pred_gen))

        # 5.3 图文匹配损失（让判别器认为假图像与文本匹配）
        loss_match = F.binary_cross_entropy_with_logits(match_pred_gen, torch.ones_like(match_pred_gen))

        # 5.4 总损失（加权求和）
        total_loss = loss_diffusion + \
                     config.LAMBDA_GAN * loss_gan + \
                     config.LAMBDA_MATCH * loss_match

        # 6. 生成器反向传播与参数更新（仅更新LoRA层）
        self.optimizer_gen.zero_grad()  # 清空梯度
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer_gen)
            self.scaler.update()  # 混合精度：更新缩放器
        else:
            total_loss.backward()
            self.optimizer_gen.step()

        # -------------------------- 新增：记录生成器损失到TensorBoard --------------------------
        self.tb_writer.add_scalar("Generator/Total_Loss", total_loss.item(), self.global_step)
        self.tb_writer.add_scalar("Generator/Diffusion_Loss", loss_diffusion.item(), self.global_step)
        self.tb_writer.add_scalar("Generator/GAN_Loss", loss_gan.item(), self.global_step)
        self.tb_writer.add_scalar("Generator/Match_Loss", loss_match.item(), self.global_step)

        # -------------------------- 新增：每 LOGGING_STEP 步可视化图像 --------------------------
        if self.global_step % config.LOGGING_STEP == 0:
            # 准备可视化数据（真实图像、生成图像、文本）
            combined_imgs, grid_titles = self._prepare_visualization_data(batch, fake_imgs)
            # 写入TensorBoard（标题含轮次和步数，方便定位）
            self.tb_writer.add_figure(
                tag=f"Train/Epoch_{self.current_epoch+1}_Step_{self.global_step}",
                figure=self._create_image_grid(combined_imgs, grid_titles),
                global_step=self.global_step
            )
        return total_loss.item(), loss_diffusion.item(), loss_gan.item(), loss_match.item()

    def train_one_epoch(self):
        """
        训练单个epoch：
        输出：训练集平均生成器损失、平均判别器损失
        """
        # 设为训练模式
        self.generator.unet.train()
        self.discriminator.train()

        # 初始化损失统计
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_diff_loss = 0.0  # 扩散损失
        total_gan_loss = 0.0   # GAN损失
        total_match_loss = 0.0 # 匹配损失

        # 进度条
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{config.EPOCHS} | Step {self.global_step}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            self.global_step += 1

            # -------------------------- 1. 训练判别器 --------------------------
            disc_loss, disc_gan_loss, disc_match_loss, disc_r1_loss = self._train_discriminator(batch)
            total_disc_loss += disc_loss

            # -------------------------- 2. 训练生成器 --------------------------
            gen_loss, gen_diff_loss, gen_gan_loss, gen_match_loss = self._train_generator(batch)
            total_gen_loss += gen_loss
            total_diff_loss += gen_diff_loss
            total_gan_loss += gen_gan_loss
            total_match_loss += gen_match_loss

            # -------------------------- 3. 日志打印 --------------------------
            if (self.global_step % config.LOGGING_STEP == 0) or (batch_idx == len(self.train_loader)-1):
                # 计算平均损失
                avg_gen_loss = total_gen_loss / (batch_idx + 1)
                avg_disc_loss = total_disc_loss / (batch_idx + 1)
                avg_diff_loss = total_diff_loss / (batch_idx + 1)
                avg_gan_loss = total_gan_loss / (batch_idx + 1)
                avg_match_loss = total_match_loss / (batch_idx + 1)

                # 打印日志
                logger.info(
                    f"Epoch {self.current_epoch+1}/{config.EPOCHS} | Step {self.global_step} | "
                    f"Gen Loss: {avg_gen_loss:.4f} | Disc Loss: {avg_disc_loss:.4f} | "
                    f"Diff Loss: {avg_diff_loss:.4f} | GAN Loss: {avg_gan_loss:.4f} | "
                    f"Match Loss: {avg_match_loss:.4f}"
                )

                # 更新进度条
                progress_bar.set_postfix({
                    "GenLoss": f"{avg_gen_loss:.4f}",
                    "DiscLoss": f"{avg_disc_loss:.4f}",
                    "DiffLoss": f"{avg_diff_loss:.4f}"
                })

            # -------------------------- 4. 按步保存检查点 --------------------------
            if CheckpointUtils.should_save_step(batch_idx, config):
                val_loss = self.validate()  # 先验证
                is_best = val_loss < self.best_val_loss
                if is_best or (not config.SAVE_BEST_ONLY):
                    self._save_checkpoint(reason=f"step_{self.global_step}", is_best=is_best)
                if is_best:
                    self.best_val_loss = val_loss
                    logger.info(f"更新最佳验证损失：{self.best_val_loss:.4f}")

        # 返回当前epoch的平均损失
        avg_gen_loss_epoch = total_gen_loss / len(self.train_loader)
        avg_disc_loss_epoch = total_disc_loss / len(self.train_loader)
        return avg_gen_loss_epoch, avg_disc_loss_epoch

    def validate(self):
        """
        验证集评估：
        输出：验证集平均生成器损失（用于判断最佳模型）
        """
        # 设为评估模式
        self.generator.unet.eval()
        self.discriminator.eval()

        total_val_loss = 0.0
        with torch.no_grad():  # 验证时不计算梯度
            val_progress = tqdm(self.val_loader, desc=f"Validating Epoch {self.current_epoch+1}")
            for batch_idx, batch in val_progress:
                # 数据预处理
                input_imgs = batch["input_image"].to(self.device)
                prompts = batch["text_prompt"]

                # 生成特征与假图像
                real_latents = self.generator.image_to_latent(input_imgs)
                text_embeds = self.generator.get_text_embeddings(prompts)
                noise = torch.randn_like(real_latents)
                # 生成「单时间步标量」（批量内所有样本用同一时间步）
                timestep_scalar = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    size=(),  # 生成标量
                    device=self.device
                ).item()  # 转为Python int
                
                # 生成「同一时间步的批量张量」（用于add_noise）
                timestep_batch = torch.tensor([timestep_scalar] * real_latents.shape[0], device=self.device)
                noisy_latents = self.noise_scheduler.add_noise(real_latents, noise, timestep_batch)

                # 生成器前向传播
                noise_pred = self.generator.predict_noise(noisy_latents, timestep_batch, text_embeds).sample
                # 调用step时传入「标量时间步」，获取去噪后的latent
                denoise_out = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep_scalar,  # 传入标量（适配单时间步step）
                    sample=noisy_latents,
                    return_dict=True  # 显式返回字典
                )
                fake_latents = denoise_out.prev_sample
                fake_imgs = self.generator.latent_to_image(fake_latents)

                # 计算验证损失（与训练时生成器损失一致）
                loss_diffusion = F.mse_loss(noise_pred, noise)
                fake_pred_gen, match_pred_gen = self.discriminator(fake_imgs, text_embeds)
                loss_gan = F.binary_cross_entropy_with_logits(fake_pred_gen, torch.ones_like(fake_pred_gen))
                loss_match = F.binary_cross_entropy_with_logits(match_pred_gen, torch.ones_like(match_pred_gen))
                batch_val_loss = (loss_diffusion + config.LAMBDA_GAN * loss_gan + config.LAMBDA_MATCH * loss_match).item()
                total_val_loss += (loss_diffusion + config.LAMBDA_GAN * loss_gan + config.LAMBDA_MATCH * loss_match).item()

                # -------------------------- 新增：验证集可视化（每轮验证只显示1次） --------------------------
                if batch_idx == 0:  # 只取第1批样本可视化，避免重复
                    combined_imgs, grid_titles = self._prepare_visualization_data(batch, fake_imgs)
                    self.tb_writer.add_figure(
                        tag=f"Val/Epoch_{self.current_epoch+1}",
                        figure=self._create_image_grid(combined_imgs, grid_titles),
                        global_step=self.global_step
                    )

                # -------------------------- 新增：记录验证损失到TensorBoard --------------------------
                self.tb_writer.add_scalar("Validation/Total_Loss", batch_val_loss, self.global_step)
                self.tb_writer.add_scalar("Validation/Diffusion_Loss", loss_diffusion.item(), self.global_step)
                self.tb_writer.add_scalar("Validation/GAN_Loss", loss_gan.item(), self.global_step)
                self.tb_writer.add_scalar("Validation/Match_Loss", loss_match.item(), self.global_step)

        # 计算平均验证损失
        avg_val_loss = total_val_loss / len(self.val_loader)
        logger.info(f"Epoch {self.current_epoch+1} 验证完成 | 平均验证损失：{avg_val_loss:.4f}")

        # 恢复训练模式
        self.generator.unet.train()
        self.discriminator.train()
        
        return avg_val_loss

    def _save_checkpoint(self, reason, is_best=False):
        """
        保存检查点：
        input：reason（保存原因，如step_100/epoch_1）、is_best（是否为最佳模型）
        """
        # 1. 基础路径（基于配置的保存路径）
        base_path = os.path.join(config.LORA_SAVE_PATH, reason)

        # 2. 保存完整检查点（含生成器、判别器参数+优化器状态）
        checkpoint = {
            # 训练状态
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            # 生成器参数（仅LoRA层）
            "generator_state_dict": self.generator.unet.state_dict(),
            "generator_optimizer": self.optimizer_gen.state_dict(),
            # 判别器参数
            "discriminator_state_dict": self.discriminator.state_dict(),
            "discriminator_optimizer": self.optimizer_disc.state_dict(),
            # 混合精度状态
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            # LoRA配置（用于后续加载）
            "lora_config": self.generator.lora_config
        }
        torch.save(checkpoint, f"{base_path}_checkpoint.pt")
        logger.info(f"完整检查点已保存：{base_path}_checkpoint.pt")

        # 3. 单独保存LoRA权重（体积小，用于推理）
        self.generator.save_lora_weights(f"{base_path}_lora")
        logger.info(f"LoRA权重已保存：{base_path}_lora")

        # 4. 保存最佳模型（单独标记）
        if is_best:
            # 最佳LoRA权重
            best_lora_path = os.path.join(config.LORA_SAVE_PATH, "best_lora")
            self.generator.save_lora_weights(best_lora_path)
            # 最佳完整检查点
            torch.save(checkpoint, os.path.join(config.LORA_SAVE_PATH, "best_checkpoint.pt"))
            logger.info(f"最佳模型已保存：{best_lora_path} | best_checkpoint.pt")

        # 5. 清理旧检查点（调用工具类方法）
        CheckpointUtils.cleanup_old_checkpoints(
            save_dir=config.LORA_SAVE_PATH,
            keep_last_n=config.SAVE_TOTAL_LIMIT
        )

    def _load_checkpoint(self, checkpoint_path):
        """
        加载检查点（用于续训）：
        input：checkpoint_path（检查点路径）
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 1. 恢复生成器参数与优化器
        self.generator.unet.load_state_dict(checkpoint["generator_state_dict"])
        self.optimizer_gen.load_state_dict(checkpoint["generator_optimizer"])

        # 2. 恢复判别器参数与优化器
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_disc.load_state_dict(checkpoint["discriminator_optimizer"])

        # 3. 恢复混合精度状态
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # 4. 恢复训练状态
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(
            f"检查点加载完成 | 恢复状态：Epoch {self.current_epoch} | "
            f"Step {self.global_step} | 最佳验证损失 {self.best_val_loss:.4f}"
        )

    def start_training(self):
        """
        启动完整训练流程：
        - 循环训练每个epoch
        - 按epoch保存检查点
        - 打印训练总结
        """
        logger.info("="*60)
        logger.info("开始 LoRA+GAN 混合训练")
        logger.info(f"训练配置：Epochs={config.EPOCHS} | Batch Size={config.BATCH_SIZE} | "
                    f"Image Size={config.IMAGE_SIZE} | Device={self.device}")
        logger.info(f"保存策略：{config.SAVE_STRATEGY} | 保留检查点数量={config.SAVE_TOTAL_LIMIT}")
        logger.info("="*60)

        # 训练循环
        for epoch in range(self.current_epoch, config.EPOCHS):
            self.current_epoch = epoch

            # 1. 训练单个epoch
            avg_gen_loss, avg_disc_loss = self.train_one_epoch()

            # 2. 验证当前epoch
            avg_val_loss = self.validate()

            # 3. 按epoch保存检查点
            if CheckpointUtils.should_save_epoch(config):
                is_best = avg_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_val_loss
                self._save_checkpoint(reason=f"epoch_{self.current_epoch+1}", is_best=is_best)

            # 4. 打印epoch总结
            logger.info(
                f"Epoch {self.current_epoch+1}/{config.EPOCHS} 总结 | "
                f"训练生成器损失：{avg_gen_loss:.4f} | 训练判别器损失：{avg_disc_loss:.4f} | "
                f"验证损失：{avg_val_loss:.4f} | 最佳验证损失：{self.best_val_loss:.4f}"
            )
            logger.info("-"*60)

        # 训练结束：保存最终模型
        self._save_checkpoint(reason="final", is_best=False)
        logger.info("="*60)
        logger.info("训练全部完成！")
        logger.info(f"最终模型保存路径：{config.LORA_SAVE_PATH}/final_lora")
        logger.info(f"最佳模型保存路径：{config.LORA_SAVE_PATH}/best_lora")
        logger.info("="*60)
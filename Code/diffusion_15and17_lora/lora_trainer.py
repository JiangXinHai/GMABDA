import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.schedulers import PNDMScheduler
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from Code.common.Config.Configs import diffusionModel_config
from Code.TwitterDataset import TwitterDataset
from Code.common.Utils import logger


class LoRATrainer:
    def __init__(self, device=None):
        # 1. 设备选择
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        self.dtype = torch.float32 
        # 2. 加载 Stable Diffusion 模型（img2img 模式）
        self.vae = AutoencoderKL.from_pretrained(
            diffusionModel_config.MODEL_NAME, 
            subfolder="vae",
            torch_dtype=self.dtype
        ).to(self.device)

        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            diffusionModel_config.MODEL_NAME,
            safety_checker=None,
            torch_dtype=self.dtype,
            use_safetensors=True
        ).to(self.device)

        # 3. LoRA 配置
        self.lora_config = LoraConfig(
            r=diffusionModel_config.LORA_R,
            lora_alpha=diffusionModel_config.LORA_ALPHA,
            target_modules=diffusionModel_config.LORA_TARGET_MODULES,
            lora_dropout=diffusionModel_config.LORA_DROPOUT,
            bias="none",
            task_type="IMAGE_2_IMAGE"
        )

        # 4. 用 LoRA 包装 UNet
        self.model = get_peft_model(self.pipeline.unet, self.lora_config)
        self.model.print_trainable_parameters()  # 打印可训练参数比例

        # 5. 使用更适合训练的调度器
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            diffusionModel_config.MODEL_NAME, 
            subfolder="scheduler"
        )

        # 6. 数据加载
        self.dataset = TwitterDataset(is_train=True)
        self.train_loader, self.val_loader = self.dataset.build_train_val_dataloaders()

        # 7. 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=diffusionModel_config.LEARNING_RATE,
            weight_decay=diffusionModel_config.WEIGHT_DECAY
        )

        # 8. 梯度缩放（用于混合精度训练）
        self.scaler = torch.cuda.amp.GradScaler() if self.device == "cuda" else None

        # 9. 输出目录
        os.makedirs(diffusionModel_config.LORA_SAVE_PATH, exist_ok=True)

        # 10. 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 11. 从检查点恢复（如果配置了）
        if diffusionModel_config.RESUME_FROM_CHECKPOINT:
            self.load_checkpoint(diffusionModel_config.RESUME_FROM_CHECKPOINT)
            logger.info(f"从检查点恢复训练: {diffusionModel_config.RESUME_FROM_CHECKPOINT}")


    def image_to_latent(self, image_tensor):
        """将 RGB 图像张量编码为 4 通道 latent"""
        with torch.no_grad():
            image_tensor = image_tensor.to(dtype=self.dtype)
            encoding = self.vae.encode(image_tensor)
            latent = encoding.latent_dist.sample()
        latent = latent * 0.18215  # Stable Diffusion 官方缩放因子
        return latent
    
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        self.current_epoch = epoch

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{diffusionModel_config.EPOCHS}")

        for step, batch in enumerate(progress_bar):
            self.global_step +=1

            # 数据到设备
            input_imgs = batch["input_image"].to(self.device)
            prompts = batch["text_prompt"]

            # 新增：图像 -> latent（4 通道）
            latents = self.image_to_latent(input_imgs)

            # 文本编码
            text_inputs = self.pipeline.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids)[0]

            # 生成噪声（现在是对 latent 加噪声）
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()

            # 添加噪声到 latent
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
               
            self.optimizer.zero_grad()
            # 混合精度训练
            if self.scaler:
                with torch.cuda.amp.autocast():
                    noise_pred = self.model(noisy_latents, timesteps, text_embeddings).sample
                    loss = F.mse_loss(noise_pred, noise)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                noise_pred = self.model(noisy_latents, timesteps, text_embeddings).sample
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                self.optimizer.step()

            # 计算损失
            total_loss += loss.item()

            # 日志记录
            if (step + 1) % diffusionModel_config.LOGGING_STEP == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(f"Epoch {epoch+1} | Step {step+1} | Global Step {self.global_step} | Loss: {avg_loss:.4f}")
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # 保存检查点
            if self._should_save_step(step):
                val_loss = self.validate() if diffusionModel_config.SAVE_BEST_ONLY else None
                is_best = val_loss < self.best_loss if val_loss is not None else False
                
                if is_best or not diffusionModel_config.SAVE_BEST_ONLY:
                    self.save_checkpoint(epoch, step, is_best=is_best, reason="step")
                    
                if is_best:
                    self.best_loss = val_loss


        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_imgs = batch["input_image"].to(self.device)
                prompts = batch["text_prompt"]

                # 新增：图像 -> latent（4 通道）
                latents = self.image_to_latent(input_imgs)

                text_inputs = self.pipeline.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=self.pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids)[0]

                # 生成噪声（对 latent 加噪声）
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
                # 添加噪声到 latent
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                # 预测噪声（输入是 noisy_latents）
                noise_pred = self.model(noisy_latents, timesteps, text_embeddings).sample

                # 混合精度验证
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        noise_pred = self.model(noisy_latents, timesteps, text_embeddings).sample
                        loss = F.mse_loss(noise_pred, noise)
                else:
                    noise_pred = self.model(noisy_latents, timesteps, text_embeddings).sample
                    loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _should_save_step(self, step):
        """判断当前步是否需要保存"""
        if diffusionModel_config.SAVE_STRATEGY == "steps":
            return (step + 1) % diffusionModel_config.SAVE_STEP == 0
        elif diffusionModel_config.SAVE_STRATEGY == "epoch":
            return False
        elif diffusionModel_config.SAVE_STRATEGY == "both":
            return (step + 1) % diffusionModel_config.SAVE_STEP == 0
        return False
    
    def _should_save_epoch(self):
        """判断当前epoch是否需要保存"""
        if diffusionModel_config.SAVE_STRATEGY in ["epoch", "both"]:
            return True
        return False

    def load_lora_weights(self, lora_path):
        """加载已训练的LoRA权重用于推理"""
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA权重文件不存在: {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        logger.info(f"成功加载LoRA权重: {lora_path}")
        return self.model
    
    def save_checkpoint(self, epoch, step, is_best=False, reason="epoch"):
        """拆分：完整检查点少存，LoRA权重正常存"""
        # 1. 基础路径
        base_path = f"{diffusionModel_config.LORA_SAVE_PATH}/{reason}_{epoch+1 if reason=='epoch' else self.global_step}"
        
        # 2. 仅在“epoch结束”或“最佳模型”时保存完整检查点（减少大文件写入）
        if reason == "epoch" or is_best:
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'lora_config': self.lora_config,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            }
            torch.save(checkpoint, f"{base_path}_checkpoint.pt")
            logger.info(f"完整检查点已保存: {base_path}_checkpoint.pt")
        
        # 3. 每次都保存LoRA权重（体积小，不影响IO）
        self.model.save_pretrained(f"{base_path}_lora")
        logger.info(f"LoRA权重已保存: {base_path}_lora")
        
        # 4. 最佳模型逻辑不变
        if is_best:
            self.model.save_pretrained(f"{diffusionModel_config.LORA_SAVE_PATH}/best_lora")
            if reason != "epoch":  # 避免重复保存最佳检查点
                torch.save(checkpoint, f"{diffusionModel_config.LORA_SAVE_PATH}/best_checkpoint.pt")
            logger.info(f"新的最佳模型已保存，验证损失: {self.best_loss:.4f}")
        
        # 5. 清理旧文件（不变）
        self.cleanup_old_checkpoints(diffusionModel_config.SAVE_TOTAL_LIMIT)

    def load_checkpoint(self, checkpoint_path):
        """加载训练检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"加载检查点: epoch={self.current_epoch}, step={self.global_step}, best_loss={self.best_loss:.4f}")
        
        return self.current_epoch, checkpoint['step']
    
    def cleanup_old_checkpoints(self, keep_last_n=5):
        """清理旧的检查点，只保留最新的几个"""
        import glob
        import re
        
        # 获取所有检查点文件
        checkpoint_files = glob.glob(f"{diffusionModel_config.LORA_SAVE_PATH}/epoch_*_checkpoint.pt")
        step_files = glob.glob(f"{diffusionModel_config.LORA_SAVE_PATH}/step_*_checkpoint.pt")
        
        # 按epoch/step排序
        def extract_number(filename):
            match = re.search(r'(epoch|step)_(\d+)', filename)
            return int(match.group(2)) if match else 0
        
        checkpoint_files.sort(key=extract_number)
        step_files.sort(key=extract_number)
        
        # 删除旧的检查点
        for files in [checkpoint_files, step_files]:
            if len(files) > keep_last_n:
                for old_file in files[:-keep_last_n]:
                    try:
                        os.remove(old_file)
                        # 同时删除对应的LoRA目录
                        lora_dir = old_file.replace('_checkpoint.pt', '_lora')
                        if os.path.exists(lora_dir):
                            import shutil
                            shutil.rmtree(lora_dir)
                        logger.info(f"已清理旧检查点: {old_file}")
                    except Exception as e:
                        logger.warning(f"清理检查点失败 {old_file}: {e}")

    def train(self):
        logger.info("开始LoRA微调训练...")
        logger.info(f"训练配置: epochs={diffusionModel_config.EPOCHS}, batch_size={diffusionModel_config.BATCH_SIZE}, lr={diffusionModel_config.LEARNING_RATE}")
        logger.info(f"保存策略: {diffusionModel_config.SAVE_STRATEGY}, 保存限制: {diffusionModel_config.SAVE_TOTAL_LIMIT}")
        for epoch in range(diffusionModel_config.EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()
            
            # 判断是否是最佳模型
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            # 保存epoch检查点
            if self._should_save_epoch():
                self.save_checkpoint(epoch, step=len(self.train_loader), 
                                is_best=is_best, reason="epoch")
            
            logger.info(f"Epoch {epoch+1} 完成 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {self.best_loss:.4f}")

        # 训练完成保存最终模型
        self.model.save_pretrained(f"{diffusionModel_config.LORA_SAVE_PATH}/final_lora")
        logger.info(f"最终模型已保存: {diffusionModel_config.LORA_SAVE_PATH}/final_lora")
        logger.info("LoRA微调完成！")
# 【最顶部添加】先验证脚本是否能正常执行
print("="*50)
print("脚本开始启动，进入基础环境检查...")
print("="*50)

# 导入模块时添加打印，定位是否是导入失败
try:
    import torch
    print("✅ torch 导入成功")
except ImportError as e:
    print(f"❌ torch 导入失败：{str(e)}")
    exit()

try:
    import os
    from PIL import Image
    from torchvision import transforms
    import numpy as np
    print("✅ 基础图像处理库导入成功")
except ImportError as e:
    print(f"❌ 基础图像处理库导入失败：{str(e)}")
    exit()

try:
    from Code.common.Utils import logger
    print("✅ 自定义 logger 导入成功")
except ImportError as e:
    print(f"❌ 自定义 logger 导入失败：{str(e)}")
    print("提示：可能是 Code.common.Utils 路径错误，或 Utils.py 中存在语法错误")
    exit()

try:
    from Code.common.Config.Configs import diffusionModel_config
    print("✅ diffusionModel_config 导入成功")
except ImportError as e:
    print(f"❌ diffusionModel_config 导入失败：{str(e)}")
    exit()

try:
    from Code.diffusion_fine_tune.generator import LoRADiffusionGenerator
    print("✅ LoRADiffusionGenerator 导入成功")
except ImportError as e:
    print(f"❌ LoRADiffusionGenerator 导入失败：{str(e)}")
    print("提示：检查 generator.py 路径是否正确，或文件内是否有语法错误/依赖缺失")
    exit()

try:
    from diffusers import DDPMScheduler
    print("✅ DDPMScheduler 导入成功")
except ImportError as e:
    print(f"❌ DDPMScheduler 导入失败：{str(e)}")
    exit()

print("="*50)
print("所有模块导入成功，开始执行测试逻辑...")
print("="*50)

def tensor_to_pil_safe(tensor, target_size):
    """
    安全地将张量转为PIL图像（彻底规避维度问题）
    1. 强制转CPU并展平多余维度
    2. 固定缩放为目标尺寸
    3. 确保通道数为3
    """
    # 1. 转CPU + 去除批量维度（只保留单图）
    tensor = tensor.detach().cpu()
    if tensor.dim() > 3:
        tensor = tensor[0]  # 取第一张图
    
    # 2. 处理通道维度（不管原始顺序，先确保通道数正确）
    if tensor.size(0) in [1, 3]:
        # 若通道在第一维（[3,H,W]或[1,H,W]），转成[H,W,C]
        tensor = tensor.permute(1, 2, 0)
    # 此时张量应为 [H,W,C]，若通道数为1则扩为3
    if tensor.size(2) == 1:
        tensor = tensor.repeat(1, 1, 3)
    # 确保通道数最终为3（防止异常情况）
    if tensor.size(2) != 3:
        tensor = tensor[:, :, :3]  # 截取前3个通道
    
    # 3. 转换到[0,255]范围（适配PIL）
    tensor = tensor.clamp(-1, 1) if tensor.max() > 1 else tensor.clamp(0, 1)
    if tensor.max() <= 1:
        tensor = (tensor * 255).byte()
    
    # 4. 转numpy再转PIL，最后强制缩放到目标尺寸
    img_np = tensor.numpy()
    pil_img = Image.fromarray(img_np)
    return pil_img.resize(target_size, Image.Resampling.LANCZOS)

def save_comparison_grid_pil(images_pil, titles, save_path, nrow=2, padding=10):
    """
    用PIL直接拼接对比图（完全不依赖张量堆叠，彻底解决维度问题）
    - images_pil: 已统一尺寸的PIL图像列表
    - padding: 图像间的空白间距（像素）
    """
    # 1. 确认所有图像尺寸一致
    img_width, img_height = images_pil[0].size
    for img in images_pil:
        assert img.size == (img_width, img_height), f"图像尺寸不一致！期望({img_width},{img_height})，实际{img.size}"
    
    # 2. 计算网格总尺寸（nrow列，自动分行）
    n_total = len(images_pil)
    n_col = nrow
    n_row = (n_total + n_col - 1) // n_col  # 向上取整
    total_width = n_col * img_width + (n_col - 1) * padding
    total_height = n_row * img_height + (n_row - 1) * padding
    
    # 3. 创建空白画布（白色背景）
    grid_img = Image.new("RGB", (total_width, total_height), color="white")
    
    # 4. 逐个粘贴图像到画布
    for idx, img in enumerate(images_pil):
        row = idx // n_col
        col = idx % n_col
        # 计算当前图像的左上角坐标
        x = col * (img_width + padding)
        y = row * (img_height + padding)
        grid_img.paste(img, (x, y))
    
    # 5. 保存最终网格图
    grid_img.save(save_path)
    logger.info(f"📊 对比图已保存：{save_path}（网格尺寸：{total_width}x{total_height}）")

def calculate_metrics_safe(original_tensor, predicted_tensor, device):
    """安全计算MSE和PSNR（展平所有维度，只看数值差异）"""
    # 1. 统一设备 + 转CPU（避免GPU计算误差）
    original = original_tensor.detach().to(device)
    predicted = predicted_tensor.detach().to(device)
    
    # 2. 展平为1维张量（彻底忽略空间维度顺序）
    original_flat = original.flatten()
    predicted_flat = predicted.flatten()
    
    # 3. 确保两个张量长度一致（取较短的长度截断）
    min_len = min(len(original_flat), len(predicted_flat))
    original_flat = original_flat[:min_len]
    predicted_flat = predicted_flat[:min_len]
    
    # 4. 转换到[0,1]范围
    original_flat = original_flat.clamp(-1, 1)
    predicted_flat = predicted_flat.clamp(-1, 1)
    original_flat = (original_flat + 1) / 2
    predicted_flat = (predicted_flat + 1) / 2
    
    # 5. 计算指标
    mse = torch.mean((original_flat - predicted_flat) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)) if mse > 0 else 100.0
    return mse.item(), psnr.item()

def test_lora_generator_with_ddpm():
    """
    测试 LoRA 生成器 + DDPMScheduler 完整流程（含完整去噪）
    核心改进：从随机时间步连续去噪到0步，确保预测图接近原图
    """
    # -------------------------- 1. 初始化核心组件 --------------------------
    # 1.1 初始化 LoRA 生成器 + 固定参数
    try:
        generator = LoRADiffusionGenerator()
        # 直接使用原始 UNet（跳过 LoRA）
        generator.unet = generator.pipeline.unet.to(generator.device)
        logger.info(f"🔍 UNet 权重验证：")
        logger.info(f"UNet 设备: {generator.pipeline.unet.device}")
        logger.info(f"UNet 第一层权重范围: {generator.pipeline.unet.conv_in.weight.min().item():.4f} ~ {generator.pipeline.unet.conv_in.weight.max().item():.4f}")
        # 正常情况：权重范围应该在 [-0.1, 0.1] 附近，不是全 0 或异常值
        model_device = generator.device
        target_size = (diffusionModel_config.IMAGE_SIZE, diffusionModel_config.IMAGE_SIZE)
        logger.info(f"✅ 生成器初始化成功：设备={model_device}，目标尺寸={target_size}")
    except Exception as e:
        logger.error(f"❌ 生成器初始化失败：{str(e)}")
        return

    # 1.2 初始化 DDPMScheduler
    try:
        num_train_timesteps = getattr(diffusionModel_config, "NUM_TRAIN_TIMESTEPS", 1000)
        beta_start = getattr(diffusionModel_config, "BETA_START", 0.0001)
        beta_end = getattr(diffusionModel_config, "BETA_END", 0.02)

        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",
            clip_sample=True,
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_inference_steps=num_train_timesteps, device=model_device)  # 适配完整去噪步长
        logger.info("✅ DDPMScheduler 初始化成功（适配完整去噪流程）")
    except Exception as e:
        logger.error(f"❌ DDPMScheduler 初始化失败：{str(e)}")
        return

    # -------------------------- 2. 准备测试数据 --------------------------
    test_image_path = "/home/jiangxinhai/GMABDA/Code/diffusion_fine_tune/test.jpg"
    output_dir = "/home/jiangxinhai/GMABDA/Code/diffusion_fine_tune/test_results"
    os.makedirs(output_dir, exist_ok=True)

    # 2.1 加载并缩放原始图片
    try:
        raw_pil = Image.open(test_image_path).convert("RGB")
        resized_original_pil = raw_pil.resize(target_size, Image.Resampling.LANCZOS)
        resized_original_pil.save(f"{output_dir}/01_original_resized.jpg")
        logger.info(f"✅ 原始图加载完成，缩放后尺寸：{resized_original_pil.size}")
    except Exception as e:
        logger.error(f"❌ 原始图加载失败：{str(e)}")
        return

    # 2.2 图像转模型输入张量
    try:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),  # [3,H,W]，范围[0,1]
            lambda x: x.unsqueeze(0)  # [1,3,H,W]
        ])
        input_tensor = transform(raw_pil).to(model_device)
        logger.info(f"✅ 模型输入张量准备完成：形状={input_tensor.shape}，设备={input_tensor.device}")
    except Exception as e:
        logger.error(f"❌ 输入张量转换失败：{str(e)}")
        return

    # 2.3 生成文本嵌入（优化：更贴近图像内容的提示词）
    try:
        # 建议根据测试图内容修改（如测试图是猫，就写"a photo of a cat, realistic"）
        test_prompt = "a photo of a cat, realistic"
        text_embeds = generator.get_text_embeddings([test_prompt]).to(model_device)
        logger.info(f"✅ 文本嵌入生成完成：形状={text_embeds.shape}")
    except Exception as e:
        logger.error(f"❌ 文本嵌入生成失败：{str(e)}")
        return

    # -------------------------- 3. 核心流程：加噪 + 完整去噪 --------------------------


    # 原图 → latent
    latent = generator.image_to_latent(input_tensor)

    # 加噪
    timestep = torch.randint(400, 401, (latent.size(0),), device=model_device, dtype=torch.int64)
    noise = torch.randn_like(latent)
    noisy_latent = scheduler.add_noise(latent, noise, timestep)

    # latent → 图像
    noisy_image = generator.latent_to_image(noisy_latent)
    noisy_pil = transforms.ToPILImage()(noisy_image[0].cpu())
    noisy_pil.save(f"{output_dir}/test_noisy.jpg")



    # 在“图像转潜空间”前，添加 VAE 测试
    try:
        # 原图 → 潜空间 → 重构图像
        latent = generator.image_to_latent(input_tensor)
        recon_image_tensor = generator.latent_to_image(latent)
        recon_pil = transforms.ToPILImage()(recon_image_tensor[0].cpu())
        recon_pil.save(f"{output_dir}/00_vae_recon.jpg")
        logger.info("✅ VAE 重构图像保存：00_vae_recon.jpg（验证 VAE 转换）")
        
        # 对比原图和重构图
        if np.array_equal(np.array(recon_pil), np.array(resized_original_pil)):
            logger.info("VAE 转换正常：重构图与原图一致")
        else:
            logger.error("VAE 转换异常：重构图与原图差异大，需检查 VAE 逻辑！")
    except Exception as e:
        logger.error(f"❌ VAE 测试失败：{str(e)}")
    # 3.1 图像转潜空间
    try:
        latent = generator.image_to_latent(input_tensor)
        logger.info(f"✅ 图像转潜空间完成：形状={latent.shape}")
    except Exception as e:
        logger.error(f"❌ 图像转潜空间失败：{str(e)}")
        return

    # 3.2 随机加噪（优化：只选低噪声阶段，去噪效果更直观）
    try:
        # 随机时间步范围：50~200（低噪声阶段，避免去噪步数过多）
        timestep = torch.randint(400, 401, (latent.size(0),), device=model_device, dtype=torch.int64)
        current_timestep_val = timestep.item()
        
        noise = torch.randn_like(latent)
        noisy_latent = scheduler.add_noise(latent, noise, timestep)

        # 保存加噪图
        noisy_image_tensor = generator.latent_to_image(noisy_latent)
        noisy_pil = tensor_to_pil_safe(noisy_image_tensor, target_size)
        noisy_pil.save(f"{output_dir}/02_noisy_image_t{current_timestep_val}.jpg")
        logger.info(f"✅ 潜空间加噪完成：时间步={current_timestep_val}（低噪声阶段），噪点图已保存")
    except Exception as e:
        logger.error(f"❌ 潜空间加噪失败：{str(e)}")
        return

    # 3.3 完整去噪流程（核心改进：从当前时间步连续降到0步）
    try:
        with torch.no_grad():
            # 1. 生成从当前时间步到0步的完整序列（倒序：current_timestep_val → 0）
            full_timesteps = torch.arange(current_timestep_val, -1, -1, device=model_device, dtype=torch.int64)
            total_steps = len(full_timesteps)
            logger.info(f"🔄 开始完整去噪：从时间步 {current_timestep_val} 到 0，共 {total_steps} 步")

            # 2. 初始化去噪 latent（从加噪后的 latent 开始）
            denoising_latent = noisy_latent.clone()

            # 3. 逐步去噪（循环执行所有时间步）
            for step_idx, t in enumerate(full_timesteps):
                # 时间步保持批量维度（匹配模型输入要求）
                t_tensor = t.unsqueeze(0)
                # 记录去噪前 latent
                prev_latent = denoising_latent.clone()
                # 模型预测当前步的噪声
                unet_output = generator.predict_noise(
                    noisy_latents=denoising_latent,
                    timesteps=t_tensor,
                    text_embeds=text_embeds
                )
                pred_noise = unet_output.sample

                # 调度器更新 latent（去噪一步）
                denoise_out = scheduler.step(
                    model_output=pred_noise,
                    timestep=t_tensor,
                    sample=denoising_latent,
                    return_dict=True
                )
                denoising_latent = denoise_out.prev_sample
                # 打印变化量
                delta = torch.norm(denoising_latent - prev_latent).item()
                logger.info(f"时间步 {t}: 变化量 = {delta:.6f}")
                logger.info(f"当前时间步: {t}, 下一时间步: {scheduler.previous_timestep(t)}")

                # 每50步打印进度（避免日志冗余）
                if (step_idx + 1) % 50 == 0 or step_idx == total_steps - 1:
                    logger.info(f"🔄 去噪进度：{step_idx + 1}/{total_steps} 步（当前时间步：{t}）")
                    denoising_img = generator.latent_to_image(denoising_latent)
                    denoising_pil = tensor_to_pil_safe(denoising_img, target_size)
                    denoising_pil.save(f"{output_dir}/03_denoise_step_t{t}.jpg")

            # 4. 去噪完成：最终 latent 对应0时间步（无噪声）
            final_denoised_latent = denoising_latent
            pred_original_latent = final_denoised_latent  # 此 latent 应接近原图潜空间
        logger.info("✅ 完整去噪流程结束！")
    except Exception as e:
        logger.error(f"❌ 完整去噪流程失败：{str(e)}")
        return

    # -------------------------- 4. 结果保存与验证 --------------------------
    # 4.2 保存最终预测图
    pred_original_image = generator.latent_to_image(pred_original_latent)
    pred_original_pil = tensor_to_pil_safe(pred_original_image, target_size)
    pred_original_pil.save(f"{output_dir}/04_final_predicted_original.jpg")

    # 4.3 计算并保存指标（此时MSE应大幅降低）
    mse, psnr = calculate_metrics_safe(input_tensor, pred_original_latent, model_device)
    logger.info(f"\n📊 最终评估指标：")
    logger.info(f"   MSE（均方误差）：{mse:.6f}（越小越接近原图）")
    logger.info(f"   PSNR（峰值信噪比）：{psnr:.2f} dB（越大越清晰）")
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write(f"测试时间步范围：{current_timestep_val} → 0\n")
        f.write(f"总去噪步数：{total_steps}\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"PSNR: {psnr:.2f} dB\n")

    # 4.4 拼接对比图（原图 + 加噪图 + 最终预测图）
    comparison_images = [
        resized_original_pil,
        noisy_pil,
        pred_original_pil
    ]
    save_comparison_grid_pil(
        images_pil=comparison_images,
        titles=["Original", f"Noisy (t={current_timestep_val})", "Final Predicted"],
        save_path=f"{output_dir}/05_comparison_grid.jpg",
        nrow=3,  # 3张图横向排列，对比更直观
        padding=10
    )

    # -------------------------- 5. 测试总结 --------------------------
    logger.info(f"\n🎉 所有测试流程完成！")
    logger.info(f"结果文件路径：{output_dir}")
    logger.info("关键文件说明：")
    logger.info("1. 01_original_resized.jpg：缩放后的原始图")
    logger.info("2. 02_noisy_image_tXXX.jpg：加噪图（测试起点）")
    logger.info("3. 03_denoise_step_tXXX.jpg：去噪中间步骤图（可选）")
    logger.info("4. 04_final_predicted_original.jpg：最终预测图（应接近原图）")
    logger.info("5. 05_comparison_grid.jpg：三图对比（直观查看去噪效果）")
    logger.info("6. metrics.txt：量化指标（MSE<0.05、PSNR>15dB为较好效果）")


if __name__ == "__main__":
    # 启动测试
    logger.info("="*60)
    logger.info("开始执行 LoRA 生成器 + DDPMScheduler 完整测试（含完整去噪）")
    logger.info("="*60)
    test_lora_generator_with_ddpm()
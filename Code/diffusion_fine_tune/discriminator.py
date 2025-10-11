import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入日志工具
from Code.common.Utils import logger
# 导入配置文件（里面有 IMAGE_SIZE、DEVICE 等参数）
from Code.common.Config.Configs import diffusionModel_config as config


class EqualLinear(nn.Module):
    """
    StyleGAN2 中使用的“等学习率线性层”
    作用：
        - 让不同输入维度的线性层具有相近的初始学习率效果
        - 权重初始化时除以 lr_mul，前向时再乘回来，从而保证梯度幅度一致
    """
    def __init__(self, in_dim, out_dim, bias=True, lr_mul=1, activation=None):
        super().__init__()
        # 权重参数（初始化时除以 lr_mul）
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        # 偏置参数（可选）
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        # 激活函数类型（目前只支持 lrelu）
        self.activation = activation
        # 缩放因子（StyleGAN2 初始化技巧）
        self.scale = (1 / torch.sqrt(torch.tensor(in_dim, dtype=torch.float32))) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        # 线性变换（带缩放）
        out = F.linear(x, self.weight * self.scale, self.bias * self.lr_mul if self.bias is not None else None)
        # 激活函数
        if self.activation == "lrelu":
            out = F.leaky_relu(out, 0.2)
        return out


class ConvLayer(nn.Sequential):
    """
    卷积层封装：Conv2d + 激活函数（LeakyReLU）
    可选下采样（AvgPool2d）
    """
    def __init__(self, in_ch, out_ch, kernel_size, down=False, activation="lrelu"):
        layers = []
        # 是否下采样
        if down:
            layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))  # 平滑下采样
        # 卷积层
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size // 2)))
        # 激活函数
        if activation == "lrelu":
            layers.append(nn.LeakyReLU(0.2))
        super().__init__(*layers)


class ResBlock(nn.Module):
    """
    残差块：
        - 两个卷积层
        - 残差连接（保证梯度流动，缓解深层网络训练困难）
        - 可选下采样
    """
    def __init__(self, in_ch, out_ch, down=False):
        super().__init__()
        # 第一个卷积（保持通道数不变）
        self.conv1 = ConvLayer(in_ch, in_ch, 3, activation="lrelu")
        # 第二个卷积（可能改变通道数，并可选下采样）
        self.conv2 = ConvLayer(in_ch, out_ch, 3, down=down, activation="lrelu")
        # 跳跃连接的卷积（如果输入输出通道数不同，用1x1卷积调整）
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # 跳跃连接的下采样（如果需要）
        self.down = nn.AvgPool2d(2) if down else nn.Identity()

    def forward(self, x):
        # 主路径
        y = self.conv1(x)
        y = self.conv2(y)
        # 残差连接
        return y + self.skip(self.down(x))


class StyleGAN2Discriminator(nn.Module):
    """
    基于 StyleGAN2 的判别器，扩展为双输出：
        1. 真实/虚假判别（real_pred）
        2. 图文匹配度判别（match_pred）
    输入：
        - 图像张量 [B, 3, H, W]（值范围[0,1]）
        - 文本嵌入 [B, 768]（CLIP 输出的文本特征）
    输出：
        - real_pred: [B, 1] 真实性分数（1表示真实，0表示虚假）
        - match_pred: [B, 1] 匹配度分数（1表示匹配，0表示不匹配）
    """
    def __init__(self, text_embed_dim=768):
        super().__init__()
        self.ch = 64  # 基础通道数
        self.resolution = config.IMAGE_SIZE  # 输入图像分辨率

        # ---------------- 文本特征投影网络 ----------------
        # CLIP 输出的文本嵌入是 768 维，这里通过 MLP 映射到 512 维
        # 目的：
        #   - 降低维度，减少计算量
        #   - 将文本特征分布映射到与图像特征兼容的空间
        self.text_proj = nn.Sequential(
            EqualLinear(text_embed_dim, 512, activation="lrelu"),
            EqualLinear(512, 512, activation="lrelu")
        )

        # ---------------- 图像特征提取网络 ----------------
        # 根据图像分辨率构建多尺度残差块
        self._build_res_blocks()

        # ---------------- 真实性判断分支 ----------------
        self.final_conv = nn.Conv2d(self.ch, self.ch // 2, 3, padding=1)
        self.final_linear = EqualLinear((self.ch // 2) * 4 * 4, 1)

        # ---------------- 图文匹配判断分支 ----------------
        # 图像特征展平后与文本特征拼接，输入全连接层
        self.match_linear = EqualLinear((self.ch // 2) * 4 * 4 + 512, 1)

        logger.info("判别器初始化完成")

    def _build_res_blocks(self):
        """
        根据图像分辨率构建网络结构：
            - 分辨率越高，需要的下采样次数越多
            - 初始通道数随分辨率增加而增加
        """
        # 不同分辨率对应的残差块数量（StyleGAN2 设计）
        num_blocks = {512: 7, 256: 6, 128: 5, 64: 4}[self.resolution]
        init_ch = self.ch * (2 ** (num_blocks - 1))

        # RGB 输入 -> 初始特征图
        self.from_rgb = nn.Conv2d(3, init_ch, 1)

        # 残差块列表
        self.blocks = nn.ModuleList()

        current_ch = init_ch
        for i in range(num_blocks):
            next_ch = current_ch // 2 if i < num_blocks - 1 else current_ch
            self.blocks.append(ResBlock(current_ch, next_ch, down=True))
            current_ch = next_ch

    def forward(self, x, text_embeds):
        """
        前向传播：
            1. 提取图像特征
            2. 投影文本特征
            3. 分别计算真实性分数和匹配度分数
        """
        # 文本特征投影：[B, 768] -> [B, 512]
        text_feat = self.text_proj(text_embeds)

        # 图像特征提取
        x = self.from_rgb(x)  # [B, C, H, W]
        for block in self.blocks:
            x = block(x)       # 每次经过残差块，空间分辨率降低一半

        # 最终卷积压缩通道数
        x = self.final_conv(x)  # [B, C//2, 4, 4]
        img_feat = x.view(x.size(0), -1)  # 展平 [B, C//2 * 4 * 4]

        # 真实性判断
        real_pred = self.final_linear(img_feat)

        # 图文匹配判断（拼接图像特征和文本特征）
        match_pred = self.match_linear(torch.cat([img_feat, text_feat], dim=1))

        return real_pred, match_pred

    def compute_r1_penalty(self, real_images, text_embeds):
        """
        R1 梯度惩罚：
            - 防止判别器过度拟合真实样本
            - 提升训练稳定性（GAN 训练常用技巧）
        """
        # 开启梯度计算
        real_images.requires_grad_(True)

        # 前向传播
        real_pred, _ = self(real_images, text_embeds)

        # 对真实样本分数求梯度
        grad = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True
        )[0]

        # 梯度平方惩罚
        grad = grad.view(grad.size(0), -1)
        r1_penalty = torch.mean(grad.pow(2))

        # 关闭梯度计算
        real_images.requires_grad_(False)

        return r1_penalty
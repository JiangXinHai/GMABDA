import numpy as np
import pandas as pd
import os
import torch
from typing import Dict, Optional, List, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from Code.common.Config.Configs import path_config, run_config, diffusionModel_config  # 新增diffusion微调配置
from Code.common.Utils import logger


class TwitterDataset(Dataset):  # 继承PyTorch的Dataset，同时保留原有功能
    """
    Twitter数据集类（支持推理+LoRA微调）
    - 推理模式：加载RunConfig.INPUT_TEXT指定的单个TSV文件（如test_15）
    - 微调模式：自动加载train_15 + dev_15（从PathConfig读取路径），生成训练样本
    """
    def __init__(self, is_train: bool = False):
        self.is_train = is_train  # 标记模式：True=微调，False=推理
        self.data: Dict[str, pd.DataFrame] = {}  # 推理模式：存储单个TSV数据

        self.processed_train_samples: Optional[List[Dict]] = None  # 微调模式：训练样本
        self.processed_val_samples: Optional[List[Dict]] = None    # 微调模式：验证样本（dev_15）
        self.image_root: str = ""  # 图像根路径（从PathConfig读取）

        # 初始化图像根路径（复用RunConfig.INPUT_IMG，训练/推理共用同一图像源）
        self._init_image_root()

        # 模式化初始化
        if self.is_train:
            self._init_train_mode()  # 微调模式：加载train+dev并预处理
        else:
            logger.info("初始化推理模式数据集（待调用load_data_text加载单文件）")

    def _init_image_root(self) -> None:
        """从PathConfig读取图像根路径（适配15/17数据集）"""
        try:
            self.image_root = path_config.DATA_PATHS_IMG_15and17[run_config.INPUT_IMG]
            if not os.path.exists(self.image_root):
                raise FileNotFoundError(f"图像根路径不存在：{self.image_root}")
            logger.info(f"图像根路径初始化完成：{self.image_root}")
        except KeyError:
            raise ValueError(f"RunConfig.INPUT_IMG={run_config.INPUT_IMG} 未在PathConfig.DATA_PATHS_IMG_15and17中定义！")

    def _init_train_mode(self) -> None:
        """微调模式初始化：加载预处理管道+train/dev数据"""
        # 1. 初始化图像预处理（匹配Stable Diffusion输入）
        self._init_image_transforms()
        # 2. 加载train_15和dev_15文本数据
        train_df, val_df = self._load_train_val_text_data()
        # 3. 对齐文本与图像，生成训练/验证样本
        self.processed_train_samples = self._align_text_image(train_df, "train")
        self.processed_val_samples = self._align_text_image(val_df, "val")
        # 4. 验证样本数量
        if len(self.processed_train_samples) == 0:
            raise ValueError("无有效训练样本！请检查文本-图像对齐逻辑或数据路径")
        logger.info(f"微调模式初始化完成：训练样本{len(self.processed_train_samples)}个，验证样本{len(self.processed_val_samples)}个")


    def _init_image_transforms(self) -> None:
        """微调模式：图像预处理管道（缩放+归一化）"""
        self.train_transform = transforms.Compose([
            transforms.Resize(
                (diffusionModel_config.IMAGE_SIZE, diffusionModel_config.IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.LANCZOS  # 高质量缩放
            ),
            # transforms.RandomHorizontalFlip(p=0.3),  # 可选：数据增强（水平翻转）
            transforms.ToTensor(),  # 转为张量：(H,W,C)→(C,H,W)，值范围[0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]
        ])
        # 验证集
        self.val_transform = transforms.Compose([
            transforms.Resize(
                (diffusionModel_config.IMAGE_SIZE, diffusionModel_config.IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_train_val_text_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """微调模式：加载 train_15 和 dev_15 文本数据（TSV 结构化数据）"""
        logger.info("开始加载微调文本数据（train_15 + dev_15）...")

        # 1. 获取路径
        try:
            train_path = path_config.DATA_PATHS_TEXT_15and17["train_15"]
            val_path = path_config.DATA_PATHS_TEXT_15and17["dev_15"]
        except KeyError as e:
            raise KeyError(f"PathConfig.DATA_PATHS_TEXT_15and17 中缺少键：{e}（需确保包含 'train_15' 和 'dev_15'）")

        # 2. 读取 TSV（指定列名，避免原始列名带空格/特殊字符）
        try:
            train_df = pd.read_csv(
                train_path,
                sep='\t',
                header=0,
                names=['index', 'label', 'image_id', 'tweet_template', 'entity']
            )
            val_df = pd.read_csv(
                val_path,
                sep='\t',
                header=0,
                names=['index', 'label', 'image_id', 'tweet_template', 'entity']
            )
        except Exception as e:
            raise RuntimeError(f"读取TSV文件失败：{e}（请检查文件路径或格式）")

        # 3. 替换 $T$ 为实体，生成真实推文
        train_df['tweet_template'] = train_df['tweet_template'].fillna("")
        train_df['entity'] = train_df['entity'].fillna("")
        train_df['text'] = train_df.apply(lambda row: row['tweet_template'].replace("$T$", row['entity']), axis=1)

        val_df['tweet_template'] = val_df['tweet_template'].fillna("")
        val_df['entity'] = val_df['entity'].fillna("")
        val_df['text'] = val_df.apply(lambda row: row['tweet_template'].replace("$T$", row['entity']), axis=1)

        # 4. 按 image_id 去重（保留第一条）
        train_df = train_df.drop_duplicates(subset=['image_id'], keep='first').reset_index(drop=True)
        val_df = val_df.drop_duplicates(subset=['image_id'], keep='first').reset_index(drop=True)

        # 5. 日志输出
        logger.info(f"成功加载并处理：train_15({len(train_df)}条)，dev_15({len(val_df)}条)")

        # 6. 返回只保留需要的列（image_id 和 text）
        return train_df[['image_id', 'text']], val_df[['image_id', 'text']]
    
    def _align_text_image(self, text_df: pd.DataFrame, split: str = "train") -> List[Dict]:
        """
        微调模式：对齐文本数据与图像数据，生成预处理后的样本
        参数：
            text_df: 文本DataFrame（train或dev）
            split: 数据划分（train/val），用于选择预处理管道
        返回：
            预处理后的样本列表（含输入图像张量、目标图像张量、文本提示）
        """
        logger.info(f"开始对齐{split}集文本与图像（共{len(text_df)}条文本）...")
        samples = []
        missing_count = 0  # 统计缺失/损坏的图像数量
        transform = self.train_transform if split == "train" else self.val_transform

        # 遍历文本数据，关联图像
        for idx, row in text_df.iterrows():
            img_filename = str(row["image_id"])  # 图像文件名（如"123.jpg"）
            text_prompt = str(row["text"])       # 文本提示（推文内容）

            # 跳过空文本或空文件名
            if not img_filename or not text_prompt:
                missing_count += 1
                logger.debug(f"第{idx}条数据跳过：图像名/文本为空")
                continue

            # 构建图像路径
            img_path = os.path.join(self.image_root, img_filename)
            if not os.path.exists(img_path):
                missing_count += 1
                logger.warning(f"第{idx}条数据跳过：图像不存在（路径：{img_path}）")
                continue

            # 加载并预处理图像（输入图像=目标图像，适用于文本-图像对齐微调）
            try:
                with Image.open(img_path).convert("RGB") as img:  # 统一转为RGB（避免灰度图）
                    input_img_tensor = transform(img)  # 输入图像（加噪声前的原始图像）
                    target_img_tensor = input_img_tensor.clone()  # 目标图像（微调的"正确结果"）

                # 保存样本
                samples.append({
                    "sample_idx": idx,
                    "input_image": input_img_tensor,  # 输入图像张量：(3, 512, 512)
                    "target_image": target_img_tensor,  # 目标图像张量：(3, 512, 512)
                    "text_prompt": text_prompt,  # 文本提示
                    "image_filename": img_filename,  # 图像文件名（日志/调试用）
                    "split": split  # 数据划分（train/val）
                })
            except Exception as e:
                missing_count += 1
                logger.error(f"第{idx}条数据跳过：图像处理失败（路径：{img_path}，错误：{e}）")

        # 输出对齐结果
        logger.info(f"{split}集对齐完成：有效样本{len(samples)}个，跳过{missing_count}个（缺失/损坏）")
        return samples

    # ---------------------- 推理模式核心方法 ----------------------
    def load_data_text(self) -> None:
        """
        推理模式：加载RunConfig.INPUT_TEXT指定的单个TSV文件（如test_15）
        """
        if run_config.INPUT_TEXT in self.data:
            logger.info(f"推理数据已加载：{run_config.INPUT_TEXT}（共{len(self.data[run_config.INPUT_TEXT])}条）")
            return

        # 从PathConfig获取推理文本路径
        try:
            text_path = path_config.DATA_PATHS_TEXT_15and17[run_config.INPUT_TEXT]
        except KeyError:
            raise ValueError(f"RunConfig.INPUT_TEXT={run_config.INPUT_TEXT} 未在PathConfig.DATA_PATHS_TEXT_15and17中定义！")

        # 读取TSV文件
        try:
            self.data[run_config.INPUT_TEXT] = pd.read_csv(text_path, sep='\t')
        except Exception as e:
            raise RuntimeError(f"读取推理TSV文件失败：{e}（路径：{text_path}）")

        logger.info(f"推理数据加载完成：{run_config.INPUT_TEXT}（共{len(self.data[run_config.INPUT_TEXT])}条）")

    def get_images(self, find_img: str) -> Optional[Image.Image]:
        """
        推理模式：获取指定文件名的图像（返回PIL.Image对象）
        参数：
            find_img: 图像文件名（如"123.jpg"，需与IMAGE_ROOT中的文件匹配）
        返回：
            PIL.Image对象（成功）/ None（失败）
        """
        img_path = os.path.join(self.image_root, find_img)
        if not os.path.exists(img_path):
            logger.warning(f"图像不存在：{img_path}")
            return None

        try:
            return Image.open(img_path).convert("RGB")  # 统一转为RGB
        except Exception as e:
            logger.error(f"打开图像失败：{img_path}（错误：{e}）")
            return None

    def save_images(self, images: List[Image.Image], img_id: str) -> None:
        """
        推理/微调模式：保存生成的图像（路径从RunConfig.OUTPUT_IMG读取）
        参数：
            images: 生成的图像列表（PIL.Image对象）
            img_id: 图像保存名（如"123_gen.jpg"）
        """
        # 从PathConfig获取输出路径
        try:
            output_dir = path_config.DATA_PATHS_IMG_15and17[run_config.OUTPUT_IMG]
        except KeyError:
            raise ValueError(f"RunConfig.OUTPUT_IMG={run_config.OUTPUT_IMG} 未在PathConfig.DATA_PATHS_IMG_15and17中定义！")

        # 创建输出目录（不存在则创建）
        os.makedirs(output_dir, exist_ok=True)
        # 补充文件扩展名（默认PNG，避免无格式文件）
        if not img_id.endswith(('.png', '.jpg', '.jpeg')):
            img_id += '.png'
        save_path = os.path.join(output_dir, img_id)

        # 保存图像（取列表第一个图像，适配diffusion模型输出）
        try:
            images[0].save(save_path)
            logger.info(f"图像保存成功：{save_path}")
        except Exception as e:
            logger.error(f"图像保存失败：{save_path}（错误：{e}）")

    # ---------------------- 微调模式核心方法（PyTorch Dataset接口） ----------------------
    def __len__(self) -> int:
        """微调模式：返回样本总数（训练+验证，或单独训练/验证）"""
        if not self.is_train:
            raise NotImplementedError("__len__仅在微调模式（is_train=True）下可用！")
        # 若需单独获取训练/验证样本数，可新增len_train()/len_val()方法
        return len(self.processed_train_samples) + len(self.processed_val_samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        微调模式：获取单个样本（PyTorch DataLoader调用）
        注：训练样本在前，验证样本在后（可根据需求调整顺序）
        """
        if not self.is_train:
            raise NotImplementedError("__getitem__仅在微调模式（is_train=True）下可用！")

        train_len = len(self.processed_train_samples)
        if idx < train_len:
            # 返回训练样本
            return self.processed_train_samples[idx]
        else:
            # 返回验证样本（索引偏移）
            return self.processed_val_samples[idx - train_len]

    def build_train_val_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        微调模式：构建训练/验证DataLoader（从DiffusionModelConfig读取超参）
        返回：
            训练DataLoader、验证DataLoader
        """
        if not self.is_train:
            raise ValueError("需初始化微调模式（is_train=True）才能构建DataLoader！")
        
        g = torch.Generator()
        g.manual_seed(diffusionModel_config.SEED)

        # 训练DataLoader（shuffle=True）
        train_loader = DataLoader(
            dataset=self.processed_train_samples,  # 直接传入训练样本列表（避免__getitem__索引判断）
            batch_size=diffusionModel_config.BATCH_SIZE,
            shuffle=True,
            num_workers=diffusionModel_config.DATA_LOADER_WORKERS,
            pin_memory=True,  # 锁存内存（加速GPU数据传输）
            drop_last=True,  # 丢弃最后一个不完整批次（避免训练报错）
            worker_init_fn=lambda worker_id: np.random.seed(diffusionModel_config.SEED + worker_id),
            generator=g
        )

        # 验证DataLoader（shuffle=False）
        val_loader = DataLoader(
            dataset=self.processed_val_samples,
            batch_size=diffusionModel_config.BATCH_SIZE,
            shuffle=False,
            num_workers=diffusionModel_config.DATA_LOADER_WORKERS,
            pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(diffusionModel_config.SEED + worker_id),
            generator=g
        )

        logger.info(f"DataLoader构建完成：训练批次数{len(train_loader)}，验证批次数{len(val_loader)}（批次大小：{diffusionModel_config.BATCH_SIZE}）")
        return train_loader, val_loader
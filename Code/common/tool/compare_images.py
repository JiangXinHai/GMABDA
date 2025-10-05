import os
import tensorflow as tf
from tensorflow.summary import create_file_writer
import numpy as np
from PIL import Image
import datetime

def load_image_original_size(file_path):
    """加载图片并保持原始尺寸，使用float16减少内存"""
    try:
        img = Image.open(file_path)
        img_array = np.array(img, dtype=np.float16)  # 用float16减少50%内存
        
        # 处理通道：灰度→RGB，RGBA→RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
            
        # 归一化到[0,1]范围
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"加载图片 {file_path} 出错: {e}")
        return None

def get_max_image_dimensions(folder_a, folder_b):
    """单独计算所有有效图片对的最大尺寸（仅读尺寸，不加载完整图片）"""
    max_height, max_width = 0, 0
    # 过滤非图片文件（可选，避免读取非图片导致错误）
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    b_files = [f for f in os.listdir(folder_b) if 
               os.path.isfile(os.path.join(folder_b, f)) and 
               f.lower().endswith(image_extensions)]
    b_files.sort()  # 按文件名排序，保证一致性
    
    for b_filename in b_files:
        if not b_filename.startswith('e'):
            continue
        
        name_with_ext = b_filename[1:]
        a_path = os.path.join(folder_a, name_with_ext)
        b_path = os.path.join(folder_b, b_filename)
        
        # 检查原始图是否存在
        if not os.path.exists(a_path):
            print(f"警告: 原始图 {name_with_ext} 不存在，跳过")
            continue
        
        # 仅读取图片尺寸（不加载像素数据，节省内存）
        try:
            with Image.open(a_path) as a_img:
                a_h, a_w = a_img.height, a_img.width
            with Image.open(b_path) as b_img:
                b_h, b_w = b_img.height, b_img.width
        except Exception as e:
            print(f"获取 {b_filename} 尺寸出错: {e}，跳过")
            continue
        
        # 更新最大尺寸（高度取两者最大值，宽度取两者之和）
        current_max_h = max(a_h, b_h)
        current_total_w = a_w + b_w
        if current_max_h > max_height:
            max_height = current_max_h
        if current_total_w > max_width:
            max_width = current_total_w
    
    return max_height, max_width

def compare_images_in_tensorboard(folder_a, folder_b, log_dir):
    """流式处理图片：单张处理+即时写入，避免全量缓存"""
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    writer = create_file_writer(log_dir)
    
    # 第一步：计算所有有效图片对的最大尺寸
    max_height, max_width = get_max_image_dimensions(folder_a, folder_b)
    if max_height == 0 or max_width == 0:
        print("没有找到有效的图片对，程序退出")
        return
    print(f"已确定最大画布尺寸：高度={max_height}px，宽度={max_width}px")
    
    # 第二步：遍历图片对并处理（仅加载当前图片，处理完释放内存）
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    b_files = [f for f in os.listdir(folder_b) if 
               os.path.isfile(os.path.join(folder_b, f)) and 
               f.lower().endswith(image_extensions)]
    b_files.sort()
    processed_count = 0  # 统计有效处理的图片对数
    
    with writer.as_default():
        for idx, b_filename in enumerate(b_files, 1):
            # 过滤不符合命名规则的文件
            if not b_filename.startswith('e'):
                print(f"警告: {b_filename} 不以'e'开头，跳过")
                continue
            
            # 构建原始图路径（变体图名去掉开头的'e'即为原始图名）
            name_with_ext = b_filename[1:]
            a_path = os.path.join(folder_a, name_with_ext)
            b_path = os.path.join(folder_b, b_filename)
            
            # 检查原始图是否存在
            if not os.path.exists(a_path):
                print(f"警告: 原始图 {name_with_ext} 不存在，跳过")
                continue
            
            # 加载当前图片对（仅缓存当前两张，处理完后自动释放）
            a_img = load_image_original_size(a_path)
            b_img = load_image_original_size(b_path)
            if a_img is None or b_img is None:
                continue
            
            # 获取当前图片的实际尺寸
            a_h, a_w = a_img.shape[0], a_img.shape[1]
            b_h, b_w = b_img.shape[0], b_img.shape[1]
            
            # 创建空白画布（白色背景，float16节省内存）
            combined_img = np.ones((max_height, max_width, 3), dtype=np.float16)
            
            # 计算图片居中放置的偏移量
            a_offset_y = (max_height - a_h) // 2  # 垂直居中
            b_offset_y = (max_height - b_h) // 2
            # 水平方向：原始图靠左，变体图接在原始图右侧
            combined_img[a_offset_y:a_offset_y+a_h, :a_w, :] = a_img
            combined_img[b_offset_y:b_offset_y+b_h, a_w:a_w+b_w, :] = b_img
            
            # 修复关键：tf.summary.image 正确参数（第一个参数是"标签名"，无tag参数）
            # 用 idx 作为 step，确保每张图独立显示；标签名包含序号，便于识别
            tf.summary.image(
                name=f"图片对比_{idx}",  # 正确参数：图片组的标签名（原tag参数错误，改为name）
                data=tf.expand_dims(combined_img, axis=0),  # 必须增加batch维度（(1, H, W, 3)）
                step=idx,  # 用序号作为step，避免图片覆盖
                description=f"原始图: {name_with_ext} | 变体图: {b_filename}"  # 图片描述（可选）
            )
            
            # 每处理10张图刷新一次日志，避免内存堆积
            if processed_count % 10 == 0:
                writer.flush()
                print(f"已临时刷新日志，当前处理进度：{processed_count} 张")
            
            # 更新统计并打印进度
            processed_count += 1
            print(f"已处理 {idx}/{len(b_files)}: 原始图={name_with_ext} | 变体图={b_filename}")
    
    # 最终刷新所有剩余日志，确保数据写入完成
    writer.flush()
    writer.close()
    
    # 输出处理结果总结
    print(f"\n=== 处理完成 ===")
    print(f"总有效图片对数量：{processed_count}")
    print(f"TensorBoard日志目录：{log_dir}")
    print(f"查看命令：tensorboard --logdir={log_dir}")
    print(f"浏览器访问：通常为 http://localhost:6006")

if __name__ == "__main__":
    # 配置路径（请根据实际情况确认）
    folder_a = "/home/jiangxinhai/GMABDA/Data/twitter2015_images"  # 原始图片文件夹
    folder_b = "/home/jiangxinhai/GMABDA/Data/twitter2015_images/generator_imgs/train_imgs"  # 变体图片文件夹
    
    # 日志目录：增加时间戳，避免覆盖历史日志
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    custom_log_dir = f"/home/jiangxinhai/GMABDA/Logs/image_comparison_{current_time}"
    
    # 启动图片对比流程
    print(f"开始图片对比任务，日志将保存到：{custom_log_dir}")
    compare_images_in_tensorboard(folder_a, folder_b, log_dir=custom_log_dir)
import os
import shutil

def merge_image_folders(folder_a, folder_b, folder_c):
    """
    合并两个文件夹中的JPG图片到第三个文件夹
    
    参数:
        folder_a: 第一个图片文件夹路径
        folder_b: 第二个图片文件夹路径
        folder_c: 目标文件夹路径
    """
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(folder_c, exist_ok=True)
    
    # 定义一个函数来复制指定文件夹中的JPG图片
    def copy_jpg_files(source_folder, dest_folder):
        # 检查源文件夹是否存在
        if not os.path.exists(source_folder):
            print(f"警告: 源文件夹 '{source_folder}' 不存在，将跳过")
            return
            
        # 遍历源文件夹中的所有文件
        for filename in os.listdir(source_folder):
            # 检查文件是否为JPG格式
            if filename.lower().endswith('.jpg'):
                source_path = os.path.join(source_folder, filename)
                dest_path = os.path.join(dest_folder, filename)
                
                # 确保是文件而不是子文件夹
                if os.path.isfile(source_path):
                    # 复制文件
                    shutil.copy2(source_path, dest_path)
                    print(f"已复制: {filename}")
    
    # 复制文件夹a中的JPG图片
    print(f"正在从 {folder_a} 复制图片...")
    copy_jpg_files(folder_a, folder_c)
    
    # 复制文件夹b中的JPG图片
    print(f"\n正在从 {folder_b} 复制图片...")
    copy_jpg_files(folder_b, folder_c)
    
    print("\n图片合并完成！")


def merge_folder_a_to_b(folder_a, folder_b):
    """
    将文件夹a中的所有JPG图片复制到文件夹b中，不进行重复检测
    
    参数:
        folder_a: 源图片文件夹路径
        folder_b: 目标图片文件夹路径
    """
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(folder_b, exist_ok=True)
    
    # 检查源文件夹是否存在
    if not os.path.exists(folder_a):
        print(f"错误: 源文件夹 '{folder_a}' 不存在，无法进行合并")
        return
    
    # 统计复制的文件数量
    copied_count = 0
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(folder_a):
        # 检查文件是否为JPG格式（不区分大小写）
        if filename.lower().endswith('.jpg'):
            source_path = os.path.join(folder_a, filename)
            dest_path = os.path.join(folder_b, filename)
            
            # 确保是文件而不是子文件夹
            if os.path.isfile(source_path):
                # 复制文件（如果已存在会直接覆盖）
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                print(f"已复制: {filename}")
    
    print(f"\n合并完成！共复制了 {copied_count} 个JPG图片到文件夹 '{folder_b}'")

if __name__ == "__main__":
    # 这里可以修改为实际的文件夹路径
    folder_a = "/home/jiangxinhai/GMABDA/Data/twitter2015_images"  # 替换为文件夹a的实际路径
    folder_b = "/home/jiangxinhai/GMABDA/Data/twitter2015_images/generator_imgs/train_imgs"  # 替换为文件夹b的实际路径
    folder_c = "/home/jiangxinhai/GMABDA/Data/twitter2015_images/15_merge"  # 替换为目标文件夹c的实际路径
    folder_train = "/home/jiangxinhai/GMABDA/Data/twitter2015_images/generator_imgs/train_imgs"
    
    # 调用函数进行合并
    merge_image_folders(folder_a, folder_b, folder_c)
    # merge_folder_a_to_b(folder_train, folder_c)
    
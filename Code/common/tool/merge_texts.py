import json
import os

def merge_json_files_no_duplicate(source_file1: str, source_file2: str, target_file: str) -> None:
    """
    合并两个JSON文件（格式为JSON对象列表，无需去重，直接拼接）
    
    参数:
        source_file1: 第一个源JSON文件路径（如 "data1.json"）
        source_file2: 第二个源JSON文件路径（如 "data2.json"）
        target_file: 合并后的目标JSON文件路径（如 "merged_data.json"）
    """
    # 读取单个JSON文件并校验格式
    def load_json_file(file_path: str) -> list:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"错误：文件 '{file_path}' 不存在")
        
        # 解析JSON并校验结构
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 确保JSON顶层是列表（符合 [{}, {}, ...] 格式）
                if not isinstance(data, list):
                    raise ValueError(f"错误：文件 '{file_path}' 不是JSON列表格式（需用 [] 包裹所有对象）")
                
                # 校验每个对象是否包含必需键（匹配你提供的JSON结构）
                required_keys = {"words", "image_id", "aspects", "opinions", "noun"}
                for idx, item in enumerate(data):
                    if not isinstance(item, dict):
                        raise ValueError(f"错误：文件 '{file_path}' 第 {idx+1} 个元素不是JSON对象")
                    missing_keys = required_keys - item.keys()
                    if missing_keys:
                        raise ValueError(f"错误：文件 '{file_path}' 第 {idx+1} 个对象缺失键：{missing_keys}")
                
                return data
        
        except json.JSONDecodeError as e:
            raise ValueError(f"错误：文件 '{file_path}' JSON格式无效（如括号不匹配、逗号错误），详情：{str(e)}")
        except Exception as e:
            raise RuntimeError(f"读取文件 '{file_path}' 失败，详情：{str(e)}")
    
    # 执行合并逻辑
    try:
        # 读取两个源文件数据
        data1 = load_json_file(source_file1)
        data2 = load_json_file(source_file2)
        
        # 直接拼接列表（无需去重）
        merged_data = data1 + data2
        
        # 写入目标文件（保留缩进，支持非ASCII字符）
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        
        # 输出合并结果
        print(f"✅ 合并完成！")
        print(f"📁 源文件1：{source_file1}（{len(data1)} 个对象）")
        print(f"📁 源文件2：{source_file2}（{len(data2)} 个对象）")
        print(f"🎯 目标文件：{target_file}（共 {len(merged_data)} 个对象）")
    
    except Exception as e:
        print(f"❌ 合并失败：{str(e)}")


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 请替换为你的实际文件路径（相对路径/绝对路径均可）
    SOURCE_FILE_train = "/home/jiangxinhai/GMABDA/Code/test/AoM/AoM-main/src/data/twitter2015/train.json"  # 第一个源JSON文件
    SOURCE_FILE_train_2 = "/home/jiangxinhai/GMABDA/Data/twitter2015/generator_texts/train_texts/train_generated.json"  # 第二个源JSON文件
    TARGET_FILE_train_final = "/home/jiangxinhai/GMABDA/Code/test/AoM/AoM-main/src/data/twitter2015_augment_v1/train.json"  # 合并后的目标文件
    
    # 调用合并函数
    merge_json_files_no_duplicate(SOURCE_FILE_train, SOURCE_FILE_train_2, TARGET_FILE_train_final)
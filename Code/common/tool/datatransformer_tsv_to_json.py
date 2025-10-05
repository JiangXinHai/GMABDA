import csv
import json
import spacy
import re
from string import punctuation


def clean_word(word):
    """清理单词中的标点符号和空白字符"""
    if word.strip() == "$T$":
        return "$T$"
    return word.strip(punctuation + ' \t\n\r').strip()


def add_spaces_around_special_tokens(text):
    """为$T$和标点左右添加空格，确保独立成词"""
    text = re.sub(r'\s*\$T\$\s*', ' $T$ ', text)
    punc_to_process = r',\. : ; ! \? @ # % & \* \( \) \[ \] \{ \} < > /'
    text = re.sub(r'(\s*)([' + punc_to_process + r'])(\s*)', r' \2 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_word_index_in_list(word, word_list, case_sensitive=False):
    """查找单词在列表中的位置"""
    for idx, w in enumerate(word_list):
        clean_w = clean_word(w)
        clean_target = clean_word(word)
        if case_sensitive:
            if clean_w == clean_target or w == word:
                return idx
        else:
            if clean_w.lower() == clean_target.lower() or w.lower() == word.lower():
                return idx
    return -1


def extract_pure_nouns(processed_text, word_list, nlp):
    """
    仅提取纯名词，过滤所有修饰成分：
    - 保留：NOUN（普通名词）、PROPN（专有名词）
    - 过滤：限定词（the, a, an）、副词、形容词、代词等
    """
    doc = nlp(processed_text)
    pure_nouns = []
    used_indices = set()
    
    # 需要排除的常见非名词词
    excluded_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 
                      'my', 'your', 'his', 'her', 'its', 'our', 'their',
                      'just', 'only', 'very', 'more', 'most'}

    # 处理名词短语，仅保留其中的纯名词
    for chunk in doc.noun_chunks:
        chunk_pure_nouns = []
        chunk_indices = []
        
        for token in chunk:
            # 仅保留名词词性，且不在排除列表中
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                clean_word(token.text).lower() not in excluded_words):
                
                idx = get_word_index_in_list(token.text, word_list)
                if idx != -1 and idx not in used_indices:
                    chunk_pure_nouns.append(word_list[idx])
                    chunk_indices.append(idx)
        
        # 只添加包含名词的短语
        if chunk_pure_nouns:
            pure_nouns.append(chunk_pure_nouns)
            for idx in chunk_indices:
                used_indices.add(idx)

    # 补充遗漏的单个名词
    for idx, word in enumerate(word_list):
        if idx in used_indices:
            continue
            
        clean_w = clean_word(word)
        if len(clean_w) < 2:
            continue
            
        # 排除常见非名词词
        if clean_w.lower() in excluded_words:
            continue
            
        # 检查是否为名词
        doc_single = nlp(word)
        for token in doc_single:
            if token.pos_ in ['NOUN', 'PROPN']:
                pure_nouns.append([word])
                used_indices.add(idx)
                break

    return pure_nouns


def convert_tsv_to_json(tsv_file, json_file, polarity_map=None):
    """TSV转JSON转换器（聚合相同image_id，仅保留纯名词）"""
    if polarity_map is None:
        polarity_map = {
            '0': 'NEU',
            '1': 'POS',
            '-1': 'NEG'
        }

    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ 成功加载spaCy模型: en_core_web_sm")
    except OSError:
        print("❌ 请先安装spaCy模型：python -m spacy download en_core_web_sm")
        return

    # -------------------------- 关键修改1：用字典按image_id聚合数据 --------------------------
    # 字典结构：key=image_id，value=该image_id对应的完整数据（含words、aspects、noun等）
    image_data_dict = {}

    with open(tsv_file, 'r', encoding='utf-8') as tsvf:
        tsv_reader = csv.reader(tsvf, delimiter='\t')
        headers = next(tsv_reader)  # 跳过表头（index	#1 Label	#2 ImageID	#3 String	#3 String）
        
        for row_num, row in enumerate(tsv_reader, 1):
            if len(row) < 5:
                print(f"⚠️  行{row_num}：字段不足（需5列），跳过")
                continue

            try:
                # 解析TSV行数据（对应表头顺序）
                index = row[0]          # 序号
                label = row[1]          # 极性标签（如1、2、0）
                image_id = row[2]       # 核心聚合键：image_id
                tweet_text = row[3].strip()  # 原始文本（含$T$）
                term_str = row[4].strip()    # $T$对应的替换术语（如Tyga、Cooperstown）
            except Exception as e:
                print(f"⚠️  行{row_num}：解析失败（{str(e)}），跳过")
                continue

            # 1. 处理文本格式（替换$T$、分割单词）
            text_with_spaces = add_spaces_around_special_tokens(tweet_text)
            processed_text = text_with_spaces.replace('$T$', term_str)  # 用术语替换$T$
            words = [w.strip() for w in processed_text.split() if w.strip()]  # 分割为单词列表
            if not words:
                print(f"⚠️  行{row_num}：处理后文本为空，跳过")
                continue

            # 2. 计算当前术语（term_str）在words中的位置（from/to）
            term_words = [w.strip() for w in term_str.split() if w.strip()]  # 术语分割为单词
            aspect_from, aspect_to = 0, 1  # 默认位置（若术语为空则用默认）
            if term_words:
                aspect_from = get_word_index_in_list(term_words[0], words)  # 术语起始位置
                if aspect_from != -1:
                    aspect_to = aspect_from + len(term_words)  # 术语结束位置（左闭右开）

            # 3. 生成当前行的aspect（术语极性信息）
            current_aspect = {
                "from": aspect_from,
                "to": aspect_to,
                "polarity": polarity_map.get(label, 'NEU'),  # 映射极性（如1→NEU）
                "term": term_words  # 术语的单词列表
            }

            # 4. 提取当前文本的纯名词（每个image_id只需提取一次，与文本对应）
            current_nouns = extract_pure_nouns(processed_text, words, nlp)

            # -------------------------- 关键修改2：按image_id聚合逻辑 --------------------------
            if image_id not in image_data_dict:
                # 若image_id首次出现：初始化该image_id的数据结构
                image_data_dict[image_id] = {
                    "words": words,                  # 文本对应的单词列表（同image_id文本应一致，若不一致取首次）
                    "image_id": image_id,            # 聚合键
                    "aspects": [current_aspect],     # 初始化aspects列表，加入当前aspect
                    "opinions": [{"term": []}],      # 固定结构（同原代码）
                    "noun": current_nouns            # 纯名词列表（同image_id文本对应，取首次）
                }
            else:
                # 若image_id已存在：仅追加aspect到aspects列表（不重复生成words和noun）
                image_data_dict[image_id]["aspects"].append(current_aspect)

            # 进度提示（每100行打印一次）
            if row_num % 100 == 0:
                print(f"🚀 已处理{row_num}行，当前聚合的image_id数量：{len(image_data_dict)}")

    # -------------------------- 关键修改3：字典转列表（最终JSON格式） --------------------------
    # 将image_data_dict的values转换为列表（JSON数组格式）
    result = list(image_data_dict.values())

    # 写入输出JSON文件
    with open(json_file, 'w', encoding='utf-8') as jsonf:
        json.dump(result, jsonf, indent=4, ensure_ascii=False)

    print(f"\n🎉 转换完成！共处理 {len(result)} 个唯一image_id（原始TSV行数：{row_num}）")
    print(f"📄 输出文件：{json_file}")
    if result:
        print(f"🔍 示例结果（首个image_id）：")
        print(f"  - image_id: {result[0]['image_id']}")
        print(f"  - aspects数量: {len(result[0]['aspects'])}")
        print(f"  - 纯名词: {result[0]['noun'][:3]}")  # 仅显示前3个名词


if __name__ == "__main__":
    # 1. 配置文件路径（根据实际路径调整）
    INPUT_TSV_train = "/home/jiangxinhai/GMABDA/Data/twitter2015/generator_texts/train_texts/paraphrased_tweets_2025-10-02.tsv"
    OUTPUT_JSON_train = "/home/jiangxinhai/GMABDA/Data/twitter2015/generator_texts/train_texts/train_generated.json"

    # 2. 极性映射配置（与TSV的#1 Label对应：0→NEG，1→NEU，2→POS）
    POLARITY_MAP = {'0': 'NEG', '1': 'NEU', '2': 'POS'}

    # 3. 测试示例文本处理效果（验证名词提取逻辑）
    test_text = "Tyga was seen changing his sexual orientation from pedophilia to messing with cougars in just a week ."
    print(f"🔍 测试文本处理效果：")
    print(f"原始文本: {test_text}")
    
    nlp_test = spacy.load("en_core_web_sm")
    test_words = test_text.split()
    test_nouns = extract_pure_nouns(test_text, test_words, nlp_test)
    print(f"提取的纯名词: {test_nouns}")  # 预期输出: [["Tyga"], ["orientation"], ["pedophilia"], ["cougars"], ["week"]]
    print("-" * 50)

    # 4. 启动TSV转JSON
    print(f"📌 开始处理训练集TSV：{INPUT_TSV_train}")
    convert_tsv_to_json(INPUT_TSV_train, OUTPUT_JSON_train, POLARITY_MAP)
import os
import json
import requests
import time
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Tuple
from TwitterDataset import TwitterDataset
from NewTwitterImgGenerator import Image2ImageGenerator
from common.Config.Configs import path_config, llm_config, run_config
from common.Utils import SentimentMapping
from tqdm import tqdm
from common.Utils import logger, add_spaces_around_special_tokens

class NewTwitterGenerator:
    def __init__(self, 
                 dataset: TwitterDataset,
        ):
        """
        初始化Twitter数据生成器
        """
        self.api_key = f"Bearer {llm_config.API_KEY}"
        self.dataset = dataset
        self.headers = {
            'Authorization': self.api_key,
            'content-type': "application/json"
        }

        # 创建输出目录（如果不存在）
        os.makedirs(path_config.DATA_PATHS_TEXT_15and17[run_config.OUTPUT_TEXT], exist_ok=True)
        self.current_data = datetime.now().strftime("%Y-%m-%d")
        self.output_tsv_path = os.path.join(
            path_config.DATA_PATHS_TEXT_15and17[run_config.OUTPUT_TEXT],
            f"paraphrased_tweets_{self.current_data}.tsv"
        )
        self.fail_id_path = "/home/jiangxinhai/GMABDA/Data/fail/failed_ids.txt"
    
    def generate_tweet_prompt(self, text: str, entity_polarity_list: List[Tuple[str, str]]) -> str:
        """Generate a tweet paraphrasing prompt based on the original text and entity sentiment polarities"""
        entity_pairs = ", ".join([f'"{entity}":"{polarity}"' for entity, polarity in entity_polarity_list])
        entity = ", ".join([f'"{entity}"' for entity, polarity in entity_polarity_list])
        
        return f"""
            Task: Paraphrase the tweet with all these rules and output the result:

            1.Original material:
                (1) TWEET: "{text}" 
                (2) ENTITY_AND_SENTIMENT:{{{entity_pairs}}}

            2.RULES:
                (1) All entities ({entity}) must be completely preserved without any addition, deletion, or modification.
                (2) Sentiment can only be negative, neutral, or positive.
                (3) Make substantial and significant changes to phrasing, sentence structure, and word choice to ensure the rewritten tweet is noticeably different from the original; avoid minor synonym substitutions. 
                (4) You may also reasonably change the emotional polarity of entities for diversity.
                (5) Enrich the tweet with relevant background knowledge while keeping it concise and on-topic.
                (6) Return only one final result with no additional explanation.
                (7) Return format: {{Rewritten Twitter content}} || {{Adjusted ENTITY_AND_SENTIMENT (same format as the original)}}.

            3.Examples for reference:  
                Original TWEET: "RT @FundsOverBuns: Tyga went from pedophile to messing with cougars all within a week."
                Original ENTITY_AND_SENTIMENT: {{"Tyga":"neutral"}}
                Output result: {{RT @FundsOverBuns: Tyga’s bad relationship shifts in a week—from pedophile links to cougars}} || {{"Tyga":"negative"}}"
        
        """
    
    def call_LLM_api(self, prompt: str) -> str:
        """调用星火大模型API生成内容
            参数:
            api_key: 大模型API密钥
            dataset: Twitter数据集处理器
            output_dir: 生成的输出目录
            api_url: 大模型API端点URL
            model: 使用的大模型
            max_retries: API调用失败时的最大重试次数
        """
        for attempt in range(llm_config.max_retries):
            try:
                body = {
                    "model": llm_config.model_name,
                    "user": "user_id",
                    "messages": [
                        {"role": "system", "content": "You are a professional text processing assistant, skilled at generating and processing text according to given instructions."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": True,
                    "tools": [
                        {
                            "type": "web_search",
                            "web_search": {
                                "enable": True,
                                "search_mode": "deep"
                            }
                        }
                    ]
                }
                full_response = ""  # 存储返回结果
                isFirstContent = True  # 首帧标识

                response = requests.post(url=llm_config.API_URL, json=body, headers=self.headers, stream=True)
                for chunks in response.iter_lines():
                    # logger.info(chunks)
                    if (chunks and '[DONE]' not in str(chunks)):
                        data_org = chunks[6:]
                        chunk = json.loads(data_org)
                        code = chunk['code']
                        if code != 0:
                            return -1
                        text = chunk['choices'][0]['delta']
                        if ('content' in text and '' != text['content']):
                            content = text["content"]
                            if (True == isFirstContent):
                                isFirstContent = False
                            full_response += content
                return full_response.strip('" ')
            
            except requests.exceptions.RequestException as e:
                wait_time = (2 ** attempt) + (attempt * 0.1)  # 指数退避
                logger.info(f"API请求失败 (尝试 {attempt + 1}/{llm_config.max_retries}): {e}")
                logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
        
        raise Exception(f"API调用失败，已达到最大重试次数 ({llm_config.max_retries})")
    
    def append_tweet_to_tsv(self, twitter_data: Dict) -> None:
        """
        单条推特数据追加到TSV文件
        参数: twitter_data - 单条推特的字典数据（需与TSV表头对应）
        """
        # 转换为DataFrame（单条数据）
        df_single = pd.DataFrame([twitter_data])
        # 判断文件是否存在：不存在则写入表头，存在则追加（不写表头）
        file_exists = os.path.exists(self.output_tsv_path)
        
        # 追加写入TSV：mode='a'（追加），header=首次写入，index=False（不写行号）
        df_single.to_csv(
            self.output_tsv_path,
            sep='\t',
            mode='a',
            header=not file_exists,
            index=False,
            encoding='utf-8'  # 避免中文/特殊字符乱码
        )
        logger.info(f"已追加新twitter文本数据twitter_data:{twitter_data}")

    def count_entity_occurrences(self, entity_list: List[Tuple[str, str]], text: str = None) -> Dict[str, int]:
        """
        统计实体的出现次数
        """
        count_dict = {}
        
        if text is None:
            # 统计原始实体列表中各实体的应有次数
            for entity, _ in entity_list:
                count_dict[entity] = count_dict.get(entity, 0) + 1
            return count_dict
        
        # 从文本中统计实体出现次数       
        processed_entities = set() 
        for entity, _ in entity_list:
            if entity in processed_entities:
                continue
            processed_entities.add(entity)
            logger.info(f"\nentity:{entity}\ntext:{text}")
            count = 0
            start = 0
            while True:
                pos = text.find(entity, start)
                if pos == -1:
                    break
                count += 1
                # 移动指针到匹配结束位置，避免重复统计
                start = pos + len(entity)
            count_dict[entity] = count
            
            # 移除已匹配的部分，防止其他实体再次匹配
            text = text.replace(entity, "", count)
        return count_dict
    
    def parse_tweet_and_entities(self, s: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        从"{文本}||{实体:情感极性, ...}"格式中提取文本和实体情感列表
        """
        # 1. 分离文本和实体部分
        parts = s.split("||")
        if len(parts) != 2:
            raise ValueError("输入格式不正确，应包含'||'分隔符")
        
        text_part = parts[0].strip()
        entity_part = parts[1].strip()
        
        # 2. 清理文本部分的外层大括号
        text = re.sub(r'^\s*{\s*|\s*}\s*$', '', text_part)
        if text.find("TWEET :"):
            text = text.replace("TWEET :", "").strip('"')
        # 3. 提取实体和情感极性
        # 支持两种引号格式："实体":"极性" 或 '实体':'极性' ,实体:"极性"
        entity_part = re.sub(r'^\s*{\s*|\s*}\s*$', '', entity_part)

        # 4. 用正则匹配 key:value 对（支持引号、逗号、分号）
        pattern = re.compile(
            r'["\']?([^"\':]+?)["\']?\s*:\s*["\']?(negative|neutral|positive)["\']?'
        )
        matches = pattern.findall(entity_part)

        entity_polarity_list = [
            (entity.strip(), polarity.strip())
            for entity, polarity in matches
        ]
        
        return text, entity_polarity_list
    
    def generate_tweets(self) -> None:
        """
        根据实体情感极性生成改写的推特
        
        """
        # 加载文本数据集
        self.dataset.load_data_text()
        df = self.dataset.data[run_config.INPUT_TEXT]
        
        logger.info(f"df:{df.index}")
        # 实例化DiffusionModel
        generator_img = Image2ImageGenerator()
        
        # 遍历DataFrame
        all_paraphrased_tweets = []
        index = df.index[0]
        total_rows = df.index[-1]
        logger.info(f"index:{index},total_rows:{total_rows}")
        processed_count = 0  # 统计成功处理的推文数量
        with tqdm(total=total_rows, desc="处理进度") as pbar:
            while index <= total_rows:
                row = df.loc[index]
                twitter_text_origin = row['#3 String']
                entity_polarity_list = []  # 新结构：List[Tuple[str, str]]
                # 记录当前twitter的起始索引（用于后续数据对齐）
                index_begin = index
                current_image_id = df.loc[index]['#2 ImageID']

                # 合并同一twitter的所有实体-情感对
                upgrade_count = 0
                while index < total_rows and df.loc[index]['#2 ImageID'] == current_image_id:
                    entity = df.loc[index]['#3 String.1']
                    polarity = SentimentMapping.num_to_label(df.loc[index]['#1 Label'])  # 映射转换
                    # 直接追加元组到列表，不做去重（保留重复实体）
                    entity_polarity_list.append( (entity, polarity) )
                    index += 1  # 移动到下一行
                    upgrade_count += 1  # 累计同组数量
                    
                # 替换占位符$T$为实际实体（生成待改写文本）
                if not entity_polarity_list:
                    logger.warning(f"【图片ID: {current_image_id}】无实体数据，跳过")
                    pbar.update(upgrade_count)
                    continue
                first_entity = entity_polarity_list[0][0]
                twitter_text_needed = twitter_text_origin.replace("$T$", first_entity)

                # 生成新推文并实时追加保存
                try:
                    logger.info(f"\n---------------------------------------begin【{index}】------------------------------------------")
                    # 1、文本改写提示词生成
                    prompt = self.generate_tweet_prompt(twitter_text_needed, entity_polarity_list)
                    logger.info(f"【第{index}条数据】生成改写指令: {prompt}...")
                    
                    # 2、请求大模型api生成改写文本
                    paraphrased_tweet = self.call_LLM_api(prompt)
                    if paraphrased_tweet == -1:
                        paraphrased_tweet = twitter_text_needed
                        logger.warning(f"【第{index}条数据】大模型返回非字符串类型")
                    # 处理空内容或纯空白
                    if not paraphrased_tweet.strip():
                        logger.warning(f"【第{index}条数据】大模型返回空内容，使用原始文本")
                        paraphrased_tweet = twitter_text_needed
                    else:
                        logger.info(f"\n【第{index}条数据】\n大模型返回结果: \n{paraphrased_tweet}")
                        paraphrased_tweet = add_spaces_around_special_tokens(paraphrased_tweet)
                        paraphrased_tweet, entity_polarity_list = self.parse_tweet_and_entities(paraphrased_tweet)
                        # 按实体长度降序排序，避免子串匹配问题
                        entity_polarity_list = sorted(
                            entity_polarity_list, 
                            key=lambda x: len(x[0]), 
                            reverse=True
                            )
                        logger.info(f"\n【第{index}条数据】\n提取文本:{paraphrased_tweet}  \n实体情感列表: {entity_polarity_list}")

                    # 3、判断文本是否包含所有实体 + 匹配实体次数 
                    try:
                        # 步骤1：统计原始实体列表中各实体的“应有次数”
                        required_counts = self.count_entity_occurrences(entity_polarity_list)  # 调用工具函数
                        # 步骤2：统计改写文本中各实体的“实际次数”
                        actual_counts = self.count_entity_occurrences(entity_polarity_list, paraphrased_tweet)
                        
                        # 步骤3：对比次数，收集不满足的实体（实际次数 < 应有次数）
                        missing_or_under_counted = []
                        for entity, required in required_counts.items():
                            actual = actual_counts[entity]
                            if actual < required:
                                missing_or_under_counted.append(f"{entity}（需{required}次，实际{actual}次）")
                        
                        # 步骤4：若存在不满足的实体，抛出异常
                        if missing_or_under_counted:
                            raise ValueError(
                                f"改写文本缺少或不足以下实体：\n"
                                f"缺失/不足详情：{', '.join(missing_or_under_counted)}\n"
                                f"改写文本：{paraphrased_tweet}\n"
                                f"原始实体列表：{[e for e, _ in entity_polarity_list]}"
                            )
                    except Exception as e:
                        # ... 原有异常处理逻辑（如记录失败ID、跳过该组）...
                        with open(self.fail_id_path, "a", encoding="utf-8") as f:
                            f.write(f"{current_image_id}\n")
                        logger.error(f"【图片ID: {current_image_id}】实体校验失败，跳过该组: {str(e)}", exc_info=True)
                        pbar.update(upgrade_count)
                        continue


                    # 4、使用改写文本和原图生成新图
                    logger.info(f"【图片ID: {current_image_id}】开始生成图片")
                    edited_imgs = generator_img.generate_from_image_and_text(
                        image=self.dataset.get_images(current_image_id),
                        prompt=f"accompanying image of \"{paraphrased_tweet}\" with undistorted, symmetric, proportional, straight lines, natural proportions."
                    )

                    # 5、判断图片生成结果是否正确
                    # 情况1：若原方法返回列表，需确保列表长度=1且元素是PIL
                    if isinstance(edited_imgs, list):
                        if len(edited_imgs) != 1:
                            raise ValueError(f"图片生成数量错误（需1张，实际{len(edited_imgs)}张）")
                    # 情况2：无效格式（如张量、路径）
                    else:
                        raise TypeError(f"图片格式错误（需PIL.Image，实际{type(edited_imgs)}）")

                    logger.info(f"【图片ID: {current_image_id}】1张图片生成成功")
                except Exception as e:
                    with open(self.fail_id_path, "a", encoding="utf-8") as f:
                        f.write(f"{current_image_id}\n")
                    logger.error(f"【图片ID: {current_image_id}】图片生成失败，跳过该组: {str(e)}", exc_info=True)
                    pbar.update(upgrade_count)
                    continue  # 图片失败，不保存任何数据
                
                # -------------------------- 图片成功后：保存1张图片 + 对应推文 --------------------------
                try:
                    self.dataset.save_images(edited_imgs, f"e{current_image_id}")

                    # 为每个实体生成一条数据并追加到TSV
                    base_text = paraphrased_tweet
                    for entity, polarity in entity_polarity_list:
                        start = base_text.find(entity)
                        end = start + len(entity)
                        base_text.replace(entity, "", 1)
                        replaced_text = paraphrased_tweet[:start] + "$T$" + paraphrased_tweet[end:]
                        twitter_data = {
                            'index': index_begin,
                            '#1 Label': SentimentMapping.label_to_num(polarity),
                            '#2 ImageID': f"e{current_image_id}",
                            '#3 String': replaced_text,
                            '#3 String.1': entity
                        }
                        # 关键：单条追加到TSV
                        self.append_tweet_to_tsv(twitter_data)
                        processed_count += 1
                        index_begin += 1  # 更新索引（对应同一图片组的不同实体）

                    logger.info(f"【图片ID: {current_image_id}】该组{len(entity_polarity_list)}条推文已保存（共享1张图片）")
                except Exception as e:
                    logger.error(f"【图片ID: {current_image_id}】保存数据出错: {str(e)}", exc_info=True)

                pbar.update(upgrade_count)  # 更新进度条
                
                # 少量测试：处理100条数据后停止（可根据需求调整）
                # if processed_count >= 50:
                #     logger.info(f"测试模式：已处理50条数据，提前停止")
                #     break

        # 最终日志：输出总处理量和文件路径
        logger.info(f"生成完成！共处理 {processed_count} 条推文，已实时追加到 {self.output_tsv_path}")

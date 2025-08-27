import os
import json
import requests
import time
import pandas as pd
from datetime import datetime
from typing import Dict
from TwitterDataset import TwitterDataset
from NewTwitterImgGenerator import Image2ImageGenerator
from common.Config.Configs import PathConfig, LLMConfig
from common.Utils import SentimentMapping
from tqdm import tqdm

class NewTwitterGenerator:
    def __init__(self, 
                 dataset: TwitterDataset,
        ):
        """
        初始化Twitter数据生成器
        """
        self.path_config = PathConfig()
        self.llm_config = LLMConfig()
        self.run_config = dataset.run_config
        self.api_key = f"Bearer {self.llm_config.API_KEY}"
        self.dataset = dataset
        self.headers = {
            'Authorization': self.api_key,
            'content-type': "application/json"
        }

        # 创建输出目录（如果不存在）
        os.makedirs(self.path_config.DATA_PATHS_TEXT_15and17[self.run_config.OUTPUT_TEXT], exist_ok=True)
    
    def generate_tweet_prompt(self, text: str, entity_polarity: Dict[str, str]) -> str:
        """Generate a tweet paraphrasing prompt based on the original text and entity sentiment polarities"""
        entity_pairs = ", ".join([f'"{entity}"—"{polarity}"' for entity, polarity in entity_polarity.items()])
        
        return f"""
        Rephrase this tweet: "{text}"
        Keep entities with sentiments: {entity_pairs}
        Rules: Only the paraphrased plain text (no explanations), same entity sentiments, similar length (slight expand if <8 words), same theme, keep retweet headers.
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
        for attempt in range(self.llm_config.max_retries):
            try:
                body = {
                    "model": self.llm_config.model_name,
                    "user": "user_id",
                    "messages": [
                        {"role": "system", "content": "你是一个专业的文本处理助手，擅长根据给定的指令生成和处理文本。"},
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

                response = requests.post(url=self.llm_config.API_URL, json=body, headers=self.headers, stream=True)
                for chunks in response.iter_lines():
                    # print(chunks)
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
                print(f"API请求失败 (尝试 {attempt + 1}/{self.llm_config.max_retries}): {e}")
                print(f"等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
        
        raise Exception(f"API调用失败，已达到最大重试次数 ({self.llm_config.max_retries})")
    
    def generate_tweets(self) -> None:
        """
        根据实体情感极性生成改写的推特
        
        """
        # 加载文本数据集
        self.dataset.load_data_text()
        df = self.dataset.data[self.run_config.INPUT_TEXT]

        # 实例化DiffusionModel
        generator_img = Image2ImageGenerator()
        
        # 遍历DataFrame
        all_paraphrased_tweets = []
        index = 0
        total_rows = len(df)
        for index in tqdm(range(total_rows), desc="处理进度", total=total_rows):

            row = df.loc[index]
            twitter_text_origin = row['#3 String']
            entity_polarity = {}
            entity = row['#3 String.1']
            polarity = SentimentMapping.num_to_label(row['#1 Label']) #映射转换
            entity_polarity[entity] = polarity
            twitter_text_needed = twitter_text_origin.replace("$T$", entity)
            # print(twitter_text_origin)
            
            index_begin = index
            while df.loc[index]['#2 ImageID'] == df.loc[index + 1]['#2 ImageID']:
                index += 1
                row = df.loc[index]
                entity = row['#3 String.1']
                polarity = SentimentMapping.num_to_label(row['#1 Label']) #映射转换
                entity_polarity[entity] = polarity
            # print(entity_polarity)
            # print(twitter_text_needed)
            
            # 生成新推文
            try:
                prompt = self.generate_tweet_prompt(twitter_text_needed, entity_polarity)
                print(prompt)
                paraphrased_tweet = self.call_LLM_api(prompt)
                if(paraphrased_tweet == -1):
                    print("请求大模型错误")
                    paraphrased_tweet = twitter_text_needed
                else :
                    print("请求大模型成功，结果已返回：", paraphrased_tweet)

                for entity, polarity in entity_polarity.items():
                    all_paraphrased_tweets.append({
                        'index': index_begin,
                        '#1 Label': SentimentMapping.label_to_num(polarity),
                        '#2 ImageID': df.loc[index_begin]['#2 ImageID'],
                        '#3 String': paraphrased_tweet.replace(entity, "$T$"),
                        '#3 String.1': entity
                    })
                    index_begin += 1
            except Exception as e:
                print(f"处理推文 {index_begin} 时出错: {e}")
                # 继续处理下一条推文

            # 生成新图片
            try:
                edited_imgs = generator_img.generate_from_image_and_text(
                    image = self.dataset.get_images(df.loc[index]['#2 ImageID']),
                    prompt = f"accompanying image of \"{paraphrased_tweet}\""
                )
                self.dataset.save_images(edited_imgs, df.loc[index]['#2 ImageID'])
            except Exception as e:
                print(f"处理图片{index}时出错：{e}")
            if index >= 3: 
                break
            index += 1

        # 保存生成的改写推文
        # 获取当前日期，格式化为年-月-日（例如：2025-08-27）
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_dir = os.path.join(self.path_config.DATA_PATHS_TEXT_15and17[self.run_config.OUTPUT_TEXT], f"paraphrased_tweets_{current_date}.tsv")
        dataframe = pd.DataFrame(all_paraphrased_tweets)
        dataframe.to_csv(output_dir, sep='\t', index=False)
        
        print(f"已成功保存 {len(all_paraphrased_tweets)} 条改写的推文到 {output_dir}")
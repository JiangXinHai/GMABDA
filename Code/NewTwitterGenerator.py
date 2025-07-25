import os
import json
import requests
import time
import pandas as pd
from typing import Dict
from TwitterDataset import TwitterDataset

class NewTwitterGenerator:
    def __init__(self, api_key: str, dataset: TwitterDataset, output_dir: str, 
                 api_url: str = "https://spark-api-open.xf-yun.com/v2/chat/completions",
                 model: str = "x1",
                 max_retries: int = 3):
        """
        初始化Twitter数据生成器
        
        参数:
            api_key: 大模型API密钥
            dataset: Twitter数据集处理器
            output_dir: 生成的输出目录
            api_url: 大模型API端点URL
            model: 使用的大模型
            max_retries: API调用失败时的最大重试次数
        """
        self.api_key = f"Bearer {api_key}"
        self.dataset = dataset
        self.output_dir = output_dir
        self.api_url = api_url
        self.model = model
        self.max_retries = max_retries
        self.headers = {
            'Authorization': self.api_key,
            'content-type': "application/json"
        }

        # 情感标签映射（0→neg，1→neu，2→pos）
        self.polarity_mapping = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        self.polarity_mapping_back = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_tweet_prompt(self, text: str, entity_polarity: Dict[str, str]) -> str:
        """Generate a tweet paraphrasing prompt based on the original text and entity sentiment polarities"""
        entity_pairs = ", ".join([f'"{entity}"—"{polarity}"' for entity, polarity in entity_polarity.items()])
        
        return f"""
        (1) Task Description: Given an original Twitter text, along with its entities and their sentiment polarities, can you generate a new Twitter text that meets the requirements in (4)?
        (2) Original Twitter Text: "{text}"
        (3) Entities and Sentiment Polarities: {entity_pairs}
        (4) Requirements: 1. The new text must be plain text and returned directly without any explanations or extra content. 
                          2. Entities and their corresponding sentiment polarities must remain unchanged. 
                          3. The text length should not be too long; if the original text is very short (fewer than 8 words), it can be appropriately expanded. 
                          4. The themes of the old and new texts should be as similar as possible. 
                          5. If there is a retweet header, it should be retained.
        """
    
    def call_LLM_api(self, prompt: str) -> str:
        """调用星火大模型API生成内容"""
        for attempt in range(self.max_retries):
            try:
                body = {
                    "model": self.model,
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

                response = requests.post(url=self.api_url, json=body, headers=self.headers, stream=True)
                print("响应头信息:", response.headers)
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
                print(f"API请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                print(f"等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
        
        raise Exception(f"API调用失败，已达到最大重试次数 ({self.max_retries})")
    
    def generate_paraphrased_tweets(self, load_data: str, output_path: str) -> None:
        """
        根据实体情感极性生成改写的推文
        
        参数:
            load_data: 指定数据集， test、train、dev
            output_path: 保存改写推文的文件路径
        """
        # 加载数据集
        self.dataset.load_data(load_data)
        # print(type(self.dataset.data[load_data]))
        # print(self.dataset.data[load_data])
        df = self.dataset.data[load_data]
        
        # 遍历DataFrame
        all_paraphrased_tweets = []
        index = 0
        total_rows = len(df)
        print(total_rows)
        while index < total_rows:
            print(index, '  :  ', index / total_rows)
            row = df.loc[index]
            twitter_text_origin = row['#3 String']
            entity_polarity = {}
            entity = row['#3 String.1']
            polarity = self.polarity_mapping.get(row['#1 Label']) #映射转换
            entity_polarity[entity] = polarity
            twitter_text_needed = twitter_text_origin.replace("$T$", entity)
            # print(twitter_text_origin)
            
            index_begin = index
            while df.loc[index]['#2 ImageID'] == df.loc[index + 1]['#2 ImageID']:
                index += 1
                row = df.loc[index]
                entity = row['#3 String.1']
                polarity = self.polarity_mapping.get(row['#1 Label']) #映射转换
                entity_polarity[entity] = polarity
            # print(entity_polarity)
            # print(twitter_text_needed)
            
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
                        '#1 Label': self.polarity_mapping_back.get(polarity),
                        '#2 ImageID': df.loc[index_begin]['#2 ImageID'],
                        '#3 String': paraphrased_tweet.replace(entity, "$T$"),
                        '#3 String.1': entity
                    })
                    index_begin += 1
            except Exception as e:
                print(f"处理推文 {index_begin} 时出错: {e}")
                # 继续处理下一条推文

            if index >= 1 : 
                # print(all_paraphrased_tweets)
                break
            index += 1

        # 保存生成的改写推文
        dataframe = pd.DataFrame(all_paraphrased_tweets)
        dataframe.to_csv(output_path, sep='\t', index=False)
        
        print(f"已成功保存 {len(all_paraphrased_tweets)} 条改写的推文到 {output_path}")
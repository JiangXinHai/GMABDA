from transformers import BartModel, BartTokenizer

local_path = "/home/jiangxinhai/GMABDA/Model/bart-base"

# 测试加载分词器
tokenizer = BartTokenizer.from_pretrained(local_path)
print("✅ 分词器加载成功，词表大小：", tokenizer.vocab_size)

# 测试加载模型
model = BartModel.from_pretrained(local_path)
print("✅ 模型加载成功，结构：", model.config)
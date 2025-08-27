import os
# 手动设置缓存路径（替换为你本地模型缓存的实际目录）
os.environ["TRANSFORMERS_CACHE"] = "~/.cache/huggingface/hub/"
# 或者直接指向模型所在的具体文件夹
# os.environ["TRANSFORMERS_CACHE"] = "/path/to/your/local/models--facebook--bart-base/"

# 验证设置是否生效
print("当前缓存路径:", os.getenv("TRANSFORMERS_CACHE"))
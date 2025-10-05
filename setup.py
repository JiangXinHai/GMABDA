from setuptools import setup, find_packages

packages = find_packages(where=".")
print("找到的包：", packages)

setup(
    name="Code",  # 你的包名，后续导入时会用到
    version="0.1.0",
    packages=find_packages(exclude=["Data", "Logs", "test", "Model"]),  # 自动识别所有含 __init__.py 的包，排除数据和测试目录
    install_requires=[
        # 核心框架
        "torch>=2.0.1",
        "diffusers>=0.23.1",
        "peft==0.10.0",
        "transformers>=4.30.2",
        "accelerate>=0.30.0",
        # 数据处理/可视化
        "pandas>=1.5.3",
        "Pillow>=11.3.0",
        "tqdm>=4.67.1",
        # 可选优化（根据实际使用添加/删除）
        "safetensors>=0.5.3",
        "xformers>=0.0.20",
        "numpy>=1.23.5",
    ],
)
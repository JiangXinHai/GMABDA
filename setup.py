from setuptools import setup, find_packages

packages = find_packages(where=".")
print("找到的包：", packages)

setup(
    name="Code",  # 你的包名，后续导入时会用到
    version="0.1.0",
    packages=find_packages(exclude=["Data", "Logs", "test", "Model"]),  # 自动识别所有含 __init__.py 的包，排除数据和测试目录
    install_requires=[],
)
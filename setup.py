from setuptools import setup, find_packages

setup(
    name="pyethnicity",
    version="0.1.0",
    description="Predict ethnicity from names using BiLSTM neural networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Based on rethnicity by Fangzhou Xie",
    url="https://github.com/0trust-rocks/pyethnicity",
    packages=find_packages(),
    package_data={"pyethnicity": ["models/*.onnx"]},
    install_requires=[
        "onnxruntime>=1.16.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
)

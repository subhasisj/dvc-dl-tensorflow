from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="src",
    version="0.0.1",
    author="subhasisj",
    description="Sample package for DVC based Deep Learning Pipeline",
    long_description=long_description,
    url="https://github.com/subhasisj/dvc-dl-tensorflow",
    author_email="subhasis.jethy@gmail.com",
    packages=["src"],
    license="BSD",
    python_requires=">=3.7",
    install_requires=[
        "dvc",
        "tensorflow-gpu==2.4.0",
        "matplotlib",
        "numpy",
        "pandas",
        "tqdm",
        "PyYAML",
        "boto3",
    ],
)

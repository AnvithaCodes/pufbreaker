from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pufbreaker",
    version="0.1.0",
    author="Anvitha Bhat A",
    description="A Python library for PUF security analysis with ML attacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnvithaCodes/pufbreaker",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.1.0",
        "jupyter>=1.0.0",
    ],
)
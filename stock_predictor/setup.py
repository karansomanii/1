from setuptools import setup, find_packages

setup(
    name="stock_predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'yfinance>=0.2.28',
        'plotly>=5.18.0',
        'streamlit>=1.28.0',
        'scikit-learn>=1.3.2',
        'tensorflow>=2.14.0',
        'nltk>=3.8.1',
        'beautifulsoup4>=4.12.2',
        'talib-binary>=0.4.24',
        'statsmodels>=0.14.0',
    ],
    author="karan somani",
    author_email="karansomanii2004@gmail.com",
    description="A sophisticated stock trading system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/karansomanii/stock_predictor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
) 
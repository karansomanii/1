from ..market_analysis import MarketSentimentAnalyzer
from ..technical_analysis import TechnicalAnalyzer
from .integrated_prediction import IntegratedPredictionSystem
from .risk_analysis import RiskAnalyzer
from .performance_tracker import PerformanceTracker
from ..ml_models_testing import MLModels

__version__ = '1.0.0'
__author__ = 'Your Name'

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_version():
    return __version__

def get_components():
    return {
        'MarketSentimentAnalyzer': MarketSentimentAnalyzer,
        'TechnicalAnalyzer': TechnicalAnalyzer,
        'IntegratedPredictionSystem': IntegratedPredictionSystem,
        'RiskAnalyzer': RiskAnalyzer,
        'PerformanceTracker': PerformanceTracker,
        'MLModels': MLModels
    }

from setuptools import setup, find_packages

setup(
    name="stock_predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'yfinance>=0.2.3',
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=0.24.2',
        'scipy>=1.7.0',
        'pandas-market-calendars>=3.2',
        'colorama>=0.4.4',
        'beautifulsoup4>=4.9.3',
        'requests>=2.25.1',
        'tqdm>=4.65.0'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Stock Movement Predictor",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock_predictor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

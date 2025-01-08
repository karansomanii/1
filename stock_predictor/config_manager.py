import json
from pathlib import Path
import os

class ConfigManager:
    def __init__(self):
        self.config_path = Path('stock_predictor/config')
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_path / 'settings.json'
        self.load_config()

    def load_config(self):
        """Load configuration from file"""
        default_config = {
            'data_settings': {
                'default_period': '3mo',
                'min_data_points': 30,
                'cache_duration': 3600,  # 1 hour in seconds
                'max_retries': 3,
                'request_timeout': 30
            },
            'model_settings': {
                'price_model': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'volatility_model': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'analysis_settings': {
                'technical': {
                    'ma_periods': [5, 10, 20, 50, 200],
                    'rsi_period': 14,
                    'macd_params': {'fast': 12, 'slow': 26, 'signal': 9}
                },
                'risk': {
                    'risk_free_rate': 0.05,
                    'market_premium': 0.06,
                    'max_position_size': 0.2
                }
            },
            'paths': {
                'data_dir': 'data',
                'models_dir': 'models',
                'cache_dir': 'cache',
                'logs_dir': 'logs'
            }
        }

        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = default_config

    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_setting(self, category, setting=None):
        """Get specific setting or category"""
        try:
            if setting is None:
                return self.config.get(category, {})
            return self.config.get(category, {}).get(setting)
        except Exception as e:
            print(f"Error getting setting: {e}")
            return None

    def update_setting(self, category, setting, value):
        """Update specific setting"""
        try:
            if category not in self.config:
                self.config[category] = {}
            self.config[category][setting] = value
            self.save_config()
            return True
        except Exception as e:
            print(f"Error updating setting: {e}")
            return False

    def create_directories(self):
        """Create necessary directories"""
        try:
            for dir_name in self.config['paths'].values():
                path = Path('stock_predictor') / dir_name
                path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directories: {e}")
            return False 
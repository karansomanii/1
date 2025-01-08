import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    def __init__(self):
        self.base_path = Path('stock_predictor/data')
        self.models_path = self.base_path / 'models'
        self.data_path = self.base_path / 'processed'
        self.cache_path = self.base_path / 'cache'
        
        # Create directories if they don't exist
        for path in [self.base_path, self.models_path, self.data_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save_trained_model(self, model, ticker, model_type):
        """Save trained model to disk"""
        try:
            filename = f"{ticker}_{model_type}_{datetime.now().strftime('%Y%m%d')}.joblib"
            filepath = self.models_path / filename
            joblib.dump(model, filepath)
            
            # Save metadata
            metadata = {
                'ticker': ticker,
                'model_type': model_type,
                'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filepath': str(filepath)
            }
            self._save_metadata(metadata, ticker)
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_trained_model(self, ticker, model_type):
        """Load trained model from disk"""
        try:
            metadata = self._load_metadata(ticker)
            if not metadata:
                return None
            
            for model_meta in metadata:
                if model_meta['model_type'] == model_type:
                    filepath = Path(model_meta['filepath'])
                    if filepath.exists():
                        return joblib.load(filepath)
            
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def save_processed_data(self, data, ticker):
        """Save processed data to disk"""
        try:
            filename = f"{ticker}_processed_{datetime.now().strftime('%Y%m%d')}.parquet"
            filepath = self.data_path / filename
            data.to_parquet(filepath)
            
            # Save metadata
            metadata = {
                'ticker': ticker,
                'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filepath': str(filepath),
                'shape': data.shape
            }
            self._save_metadata(metadata, ticker, data_type='processed')
            
            return True
        except Exception as e:
            print(f"Error saving processed data: {e}")
            return False
    
    def load_processed_data(self, ticker):
        """Load processed data from disk"""
        try:
            metadata = self._load_metadata(ticker, data_type='processed')
            if not metadata:
                return None
            
            # Get most recent data
            latest_meta = max(metadata, key=lambda x: x['date_processed'])
            filepath = Path(latest_meta['filepath'])
            
            if filepath.exists():
                return pd.read_parquet(filepath)
            
            return None
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None
    
    def cache_results(self, results, ticker):
        """Cache analysis results"""
        try:
            filename = f"{ticker}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.cache_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error caching results: {e}")
            return False
    
    def get_cached_results(self, ticker):
        """Get cached analysis results"""
        try:
            pattern = f"{ticker}_results_*.json"
            files = list(self.cache_path.glob(pattern))
            
            if not files:
                return None
            
            # Get most recent cache file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cached results: {e}")
            return None
    
    def _save_metadata(self, metadata, ticker, data_type='model'):
        """Save metadata for models or processed data"""
        try:
            metadata_file = self.base_path / f"{ticker}_{data_type}_metadata.json"
            
            existing_metadata = []
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            
            existing_metadata.append(metadata)
            
            with open(metadata_file, 'w') as f:
                json.dump(existing_metadata, f, indent=4)
                
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def _load_metadata(self, ticker, data_type='model'):
        """Load metadata for models or processed data"""
        try:
            metadata_file = self.base_path / f"{ticker}_{data_type}_metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            
            return []
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return [] 
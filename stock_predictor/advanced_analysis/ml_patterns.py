import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import talib
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors

class PatternRecognitionML:
    def __init__(self):
        self.logger = Logger()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.lstm_model = self._build_lstm_model()
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def _build_lstm_model(self):
        """Build LSTM model for pattern recognition"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(100, 6)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @handle_errors
    def train_models(self, historical_data):
        """Train all ML models"""
        try:
            # Prepare features
            X, y = self._prepare_training_data(historical_data)
            
            # Train Random Forest
            self.rf_model.fit(X, y)
            
            # Train Gradient Boosting
            self.gb_model.fit(X, y)
            
            # Train LSTM
            X_lstm = self._prepare_lstm_data(historical_data)
            y_lstm = y[-len(X_lstm):]
            self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            return False
    
    def _prepare_training_data(self, data):
        """Prepare features for traditional ML models"""
        try:
            features = []
            labels = []
            
            # Calculate technical indicators
            for i in range(100, len(data)):
                window = data.iloc[i-100:i]
                
                # Price action features
                price_features = self._extract_price_features(window)
                
                # Technical indicators
                tech_features = self._extract_technical_features(window)
                
                # Volume features
                vol_features = self._extract_volume_features(window)
                
                # Combine features
                all_features = np.concatenate([
                    price_features,
                    tech_features,
                    vol_features
                ])
                
                features.append(all_features)
                
                # Label: 1 if price increases by 2% within next 5 days
                future_return = (
                    data['Close'].iloc[i:i+5].max() - data['Close'].iloc[i]
                ) / data['Close'].iloc[i]
                labels.append(1 if future_return > 0.02 else 0)
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            self.logger.error(f"Data preparation error: {e}")
            return None, None
    
    def _prepare_lstm_data(self, data):
        """Prepare data for LSTM model"""
        try:
            sequence_length = 100
            features = []
            
            for i in range(sequence_length, len(data)):
                sequence = data.iloc[i-sequence_length:i]
                
                # Normalize sequence
                normalized = self.scaler.fit_transform(sequence[[
                    'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'
                ]])
                
                features.append(normalized)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"LSTM data preparation error: {e}")
            return None
    
    @handle_errors
    def predict_pattern(self, data):
        """Predict patterns using all models"""
        try:
            if not self.is_trained:
                return None
            
            # Prepare features
            X = self._prepare_prediction_features(data)
            X_lstm = self._prepare_lstm_data(data)
            
            # Get predictions from all models
            rf_pred = self.rf_model.predict_proba(X)[-1][1]
            gb_pred = self.gb_model.predict_proba(X)[-1][1]
            lstm_pred = self.lstm_model.predict(X_lstm)[-1][0]
            
            # Ensemble predictions
            ensemble_pred = (rf_pred + gb_pred + lstm_pred) / 3
            
            # Identify specific patterns
            patterns = self._identify_specific_patterns(data)
            
            return {
                'probability': float(ensemble_pred),
                'patterns': patterns,
                'model_predictions': {
                    'random_forest': float(rf_pred),
                    'gradient_boosting': float(gb_pred),
                    'lstm': float(lstm_pred)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pattern prediction error: {e}")
            return None
    
    def _identify_specific_patterns(self, data):
        """Identify specific chart patterns"""
        try:
            patterns = {}
            
            # Price action patterns
            patterns['trend_reversal'] = self._check_trend_reversal(data)
            patterns['breakout'] = self._check_breakout(data)
            patterns['consolidation'] = self._check_consolidation(data)
            
            # Candlestick patterns
            patterns['candlestick'] = self._check_candlestick_patterns(data)
            
            # Volume patterns
            patterns['volume'] = self._check_volume_patterns(data)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Pattern identification error: {e}")
            return {}
    
    def _check_trend_reversal(self, data):
        """Check for trend reversal patterns"""
        try:
            close = data['Close'].values
            sma20 = talib.SMA(close, timeperiod=20)
            sma50 = talib.SMA(close, timeperiod=50)
            
            # Check for bullish reversal
            bullish_reversal = (
                sma20[-1] > sma20[-2] and
                sma20[-2] < sma20[-3] and
                close[-1] > sma50[-1]
            )
            
            # Check for bearish reversal
            bearish_reversal = (
                sma20[-1] < sma20[-2] and
                sma20[-2] > sma20[-3] and
                close[-1] < sma50[-1]
            )
            
            return {
                'bullish_reversal': bullish_reversal,
                'bearish_reversal': bearish_reversal
            }
            
        except Exception as e:
            self.logger.error(f"Trend reversal check error: {e}")
            return {}
    
    def _check_breakout(self, data):
        """Check for breakout patterns"""
        try:
            close = data['Close'].values
            volume = data['Volume'].values
            
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            
            # Breakout conditions
            bullish_breakout = (
                close[-1] > upper[-1] and
                volume[-1] > volume[-2] * 1.5
            )
            
            bearish_breakout = (
                close[-1] < lower[-1] and
                volume[-1] > volume[-2] * 1.5
            )
            
            return {
                'bullish_breakout': bullish_breakout,
                'bearish_breakout': bearish_breakout
            }
            
        except Exception as e:
            self.logger.error(f"Breakout check error: {e}")
            return {}

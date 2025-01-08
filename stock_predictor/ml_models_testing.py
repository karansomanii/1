from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class MLModels:
    def __init__(self):
        self.sentiment_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.volatility_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        self.is_trained = False 
from .market_analysis import MarketSentimentAnalyzer
from .technical_analysis import TechnicalAnalyzer
from .integrated_prediction import IntegratedPredictionSystem
from .risk_analysis import RiskAnalyzer
from .performance_tracker import PerformanceTracker
from .ml_models_testing import MLModels

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from colorama import init, Fore, Back, Style
from tqdm import tqdm
import sys
import io
import time
import random
import warnings
warnings.filterwarnings('ignore')

class AdvancedMarketPredictor:
    def __init__(self):
        self.prediction_system = IntegratedPredictionSystem()
        self.risk_analyzer = RiskAnalyzer()
        self.performance_tracker = PerformanceTracker()
        
    def analyze_stock(self, ticker, data):
        """Complete stock analysis and prediction"""
        prediction = self.prediction_system.predict_movement(ticker, data)
        if prediction:
            risk_analysis = self.risk_analyzer.analyze_risk(ticker, data, prediction)
            prediction['risk_analysis'] = risk_analysis
            
            # Track prediction
            self.performance_tracker.track_prediction(ticker, prediction)
            
            # Add backtest results
            prediction['backtest'] = self.performance_tracker.backtest_strategy(ticker)
            
            # Add performance metrics
            prediction['performance'] = self.performance_tracker.analyze_performance()
            
        return prediction

    def download_data(self, ticker, period="3mo", max_retries=3):
        """Download data with retries and rate limiting"""
        print(f"\n{Fore.YELLOW}Downloading data for {ticker}...{Style.RESET_ALL}")
        
        for attempt in range(max_retries):
            try:
                # Suppress yfinance output
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                # Add random delay between attempts
                if attempt > 0:
                    time.sleep(random.uniform(2, 5))
                
                data = yf.download(ticker, period=period, progress=False)
                sys.stdout = old_stdout
                
                if len(data) > 0:
                    return data
                
            except Exception as e:
                sys.stdout = old_stdout
                print(f"{Fore.YELLOW}Attempt {attempt + 1} failed: {e}{Style.RESET_ALL}")
        
        print(f"{Fore.RED}Failed to download data after {max_retries} attempts{Style.RESET_ALL}")
        return None

def get_next_trading_day(ticker):
    """Get next trading day based on exchange"""
    try:
        is_indian = ticker.endswith('.NS')
        calendar = mcal.get_calendar('NSE' if is_indian else 'NYSE')
        
        today = datetime.now().date()
        schedule = calendar.schedule(
            start_date=today,
            end_date=today + timedelta(days=7)
        )
        
        next_trading_day = schedule.index[0]
        return next_trading_day.strftime('%Y-%m-%d')
        
    except Exception as e:
        print(f"Error determining next trading day: {e}")
        return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

def format_prediction_output(prediction, ticker, next_trading_day):
    """Format prediction results for display"""
    print(f"\n{Back.WHITE}{Fore.BLACK}================ Analysis Results for {ticker} ================={Style.RESET_ALL}")
    
    # Movement prediction
    direction = prediction['direction']
    color = Fore.GREEN if direction == 'upward' else Fore.RED if direction == 'downward' else Fore.YELLOW
    print(f"\n{Fore.CYAN}Predicted Movement ({next_trading_day}):{Style.RESET_ALL}")
    print(f"Direction: {color}{direction.upper()}{Style.RESET_ALL}")
    print(f"Expected Change: {color}{prediction['predicted_change']*100:.2f}%{Style.RESET_ALL}")
    print(f"Predicted Volatility: {prediction['predicted_volatility']*100:.2f}%")
    print(f"Confidence Level: {prediction['confidence']*100:.2f}%")
    
    # Key price levels
    levels = prediction['analysis']['key_levels']
    print(f"\n{Fore.CYAN}Key Price Levels:{Style.RESET_ALL}")
    print(f"Current Price: ₹{levels['current_price']:.2f}")
    print(f"Target Price: ₹{levels['predicted_price']:.2f}")
    if levels['nearest_resistance']:
        print(f"Nearest Resistance: ₹{levels['nearest_resistance']:.2f}")
    if levels['nearest_support']:
        print(f"Nearest Support: ₹{levels['nearest_support']:.2f}")
    
    # Market Sentiment
    sentiment = prediction['analysis']['sentiment']
    print(f"\n{Fore.CYAN}Market Sentiment Analysis:{Style.RESET_ALL}")
    components = sentiment['sentiment_components']
    print(f"Overall Sentiment: {_format_sentiment(sentiment['overall_sentiment'])}")
    print(f"Global Markets: {_format_sentiment(components['global'])}")
    print(f"Fundamentals: {_format_sentiment(components['fundamental'])}")
    print(f"Sector: {_format_sentiment(components['sector'])}")
    
    # Technical Analysis
    technical = prediction['analysis']['technical']
    print(f"\n{Fore.CYAN}Technical Indicators:{Style.RESET_ALL}")
    
    # Momentum
    momentum = technical['momentum']
    print(f"RSI: {momentum.get('RSI', 0):.2f}")
    macd = momentum.get('MACD', {})
    if macd:
        print(f"MACD Histogram: {macd.get('histogram', 0):.2f}")
    
    # Volatility
    volatility = technical['volatility']
    print(f"Daily Volatility: {volatility.get('daily_volatility', 0)*100:.2f}%")
    print(f"ATR: {volatility.get('atr', 0):.2f}")
    
    # Patterns
    patterns = technical['patterns']
    if patterns:
        print(f"\n{Fore.CYAN}Detected Patterns:{Style.RESET_ALL}")
        for pattern, details in patterns.items():
            print(f"• {pattern.replace('_', ' ').title()}")
            print(f"  Confidence: {details['confidence']*100:.2f}%")
            print(f"  Target: ₹{details['price_target']:.2f}")
    
    # Add Risk Analysis Section
    risk_analysis = prediction.get('risk_analysis', {})
    if risk_analysis:
        print(f"\n{Back.WHITE}{Fore.BLACK}================ Risk Analysis ================={Style.RESET_ALL}")
        
        # Risk Metrics
        metrics = risk_analysis['risk_metrics']
        print(f"\n{Fore.CYAN}Risk Metrics:{Style.RESET_ALL}")
        print(f"Volatility (Annual): {metrics['volatility']*100:.2f}%")
        print(f"Value at Risk (95%): {metrics['var_95']*100:.2f}%")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Risk Score: {metrics['risk_score']:.2f}/100")
        
        # Trade Setup
        setup = risk_analysis['trade_setup']
        print(f"\n{Fore.CYAN}Trade Setup:{Style.RESET_ALL}")
        print(f"Position Type: {setup['position_type']}")
        print(f"Entry Price: ₹{setup['entry']['price']:.2f} ({setup['entry']['type']})")
        print(f"Stop Loss: ₹{setup['stop_loss']:.2f}")
        print(f"Take Profit Levels:")
        for i, tp in enumerate(setup['take_profits'], 1):
            print(f"  TP{i}: ₹{tp:.2f}")
        print(f"Risk-Reward Ratio: {setup['risk_reward_ratio']:.2f}")
        print(f"Recommended Position Size: {setup['recommended_size']*100:.2f}% of portfolio")
        
        # Price Targets
        targets = risk_analysis['targets']
        print(f"\n{Fore.CYAN}Price Targets:{Style.RESET_ALL}")
        
        print("Short Term (1-2 weeks):")
        st = targets['short_term']
        print(f"  Target: ₹{st['target']:.2f}")
        print(f"  Probability: {st['probability']*100:.2f}%")
        
        print("\nMedium Term (1-2 months):")
        mt = targets['medium_term']
        print(f"  Target: ₹{mt['target']:.2f}")
        print(f"  Probability: {mt['probability']*100:.2f}%")
        
        print("\nVolatility Adjusted Target:")
        va = targets['volatility_adjusted']
        print(f"  Target: ₹{va['target']:.2f}")
        print(f"  Probability: {va['probability']*100:.2f}%")
        
        # Trading Summary
        print(f"\n{Fore.CYAN}Trading Summary:{Style.RESET_ALL}")
        position_size = risk_analysis['position_size']
        print(f"Risk Factor: {position_size['risk_factor']:.2f}")
        print(f"Confidence Factor: {position_size['confidence_factor']:.2f}")
        print(f"Volatility Factor: {position_size['volatility_factor']:.2f}")
        
        # Trading Recommendation
        setup_type = setup['position_type']
        confidence = prediction['confidence']
        risk_score = metrics['risk_score']
        
        print(f"\n{Fore.CYAN}Trading Recommendation:{Style.RESET_ALL}")
        if confidence > 0.7 and risk_score < 70:
            color = Fore.GREEN if setup_type == 'LONG' else Fore.RED
            print(f"{color}Strong {setup_type} opportunity{Style.RESET_ALL}")
            print(f"• Entry around ₹{setup['entry']['price']:.2f}")
            print(f"• Stop Loss at ₹{setup['stop_loss']:.2f}")
            print(f"• Initial Target: ₹{setup['take_profits'][0]:.2f}")
            print(f"• Position Size: {setup['recommended_size']*100:.2f}%")
        elif confidence > 0.5 and risk_score < 80:
            print(f"{Fore.YELLOW}Moderate {setup_type} opportunity with caution{Style.RESET_ALL}")
            print("• Consider smaller position size")
            print("• Use tight stops")
        else:
            print(f"{Fore.RED}High risk - Consider waiting for better setup{Style.RESET_ALL}")
            print("• Market conditions unfavorable")
            print("• Risk metrics elevated")

    # Add Performance Analysis Section
    print(f"\n{Back.WHITE}{Fore.BLACK}================ Performance Analysis ================={Style.RESET_ALL}")
    
    # Historical Performance
    performance = prediction.get('performance')
    if performance:
        print(f"\n{Fore.CYAN}Prediction Performance (Last 30 Days):{Style.RESET_ALL}")
        print(f"Total Predictions: {performance['total_predictions']}")
        print(f"Accuracy: {performance['accuracy']*100:.2f}%")
        print(f"Precision: {performance['precision']*100:.2f}%")
        print(f"Recall: {performance['recall']*100:.2f}%")
        print(f"Profitable Trades: {performance['profitable_trades']}")
        print(f"Average Return: {performance['avg_return']*100:.2f}%")
        print(f"Average Confidence: {performance['avg_confidence']*100:.2f}%")
        print(f"Average Risk Score: {performance['avg_risk_score']:.2f}")
    
    # Backtest Results
    backtest = prediction.get('backtest')
    if backtest:
        print(f"\n{Fore.CYAN}Backtest Results:{Style.RESET_ALL}")
        print(f"Total Trades: {backtest['total_trades']}")
        print(f"Win Rate: {backtest['win_rate']*100:.2f}%")
        print(f"Average Return: {backtest['avg_return']*100:.2f}%")
        print(f"Max Drawdown: {backtest['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
        print(f"Profit Factor: {backtest['profit_factor']:.2f}")
        
        # Performance Rating
        rating = _calculate_performance_rating(backtest)
        print(f"\n{Fore.CYAN}Strategy Rating: {_format_rating(rating)}")

def _format_sentiment(value):
    """Format sentiment value with color"""
    if value > 0.2:
        return f"{Fore.GREEN}Bullish ({value:.2f}){Style.RESET_ALL}"
    elif value < -0.2:
        return f"{Fore.RED}Bearish ({value:.2f}){Style.RESET_ALL}"
    else:
        return f"{Fore.YELLOW}Neutral ({value:.2f}){Style.RESET_ALL}"

def _calculate_performance_rating(backtest):
    """Calculate overall performance rating"""
    try:
        # Weight different metrics
        win_rate_score = min(100, backtest['win_rate'] * 100)
        return_score = min(100, backtest['avg_return'] * 1000)  # Scale returns
        drawdown_score = max(0, 100 - backtest['max_drawdown'] * 100)
        sharpe_score = min(100, backtest['sharpe_ratio'] * 25)  # Scale Sharpe
        
        # Calculate weighted average
        rating = (
            win_rate_score * 0.3 +
            return_score * 0.3 +
            drawdown_score * 0.2 +
            sharpe_score * 0.2
        )
        
        return rating
        
    except:
        return 50  # Default middle rating

def _format_rating(rating):
    """Format performance rating with color and stars"""
    if rating >= 80:
        return f"{Fore.GREEN}★★★★★ ({rating:.1f}/100) Excellent{Style.RESET_ALL}"
    elif rating >= 60:
        return f"{Fore.LIGHTGREEN_EX}★★★★☆ ({rating:.1f}/100) Good{Style.RESET_ALL}"
    elif rating >= 40:
        return f"{Fore.YELLOW}★★★☆☆ ({rating:.1f}/100) Average{Style.RESET_ALL}"
    elif rating >= 20:
        return f"{Fore.LIGHTRED_EX}★★☆☆☆ ({rating:.1f}/100) Below Average{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}★☆☆☆☆ ({rating:.1f}/100) Poor{Style.RESET_ALL}"

def main():
    """Main execution loop"""
    init()  # Initialize colorama
    
    print(f"\n{Back.WHITE}{Fore.BLACK}================================================{Style.RESET_ALL}")
    print(f"{Back.WHITE}{Fore.BLACK}        Advanced Stock Movement Predictor         {Style.RESET_ALL}")
    print(f"{Back.WHITE}{Fore.BLACK}================================================{Style.RESET_ALL}")
    
    predictor = AdvancedMarketPredictor()
    
    while True:
        try:
            print(f"\n{Fore.CYAN}Enter stock ticker symbol (or 'quit' to exit):{Style.RESET_ALL} ", end='')
            ticker = input().upper().strip()
            
            if ticker == 'QUIT':
                # Show final performance summary
                performance = predictor.performance_tracker.analyze_performance()
                if performance:
                    print(f"\n{Back.WHITE}{Fore.BLACK}Final Performance Summary{Style.RESET_ALL}")
                    print(f"Total Predictions: {performance['total_predictions']}")
                    print(f"Overall Accuracy: {performance['accuracy']*100:.2f}%")
                    print(f"Average Return: {performance['avg_return']*100:.2f}%")
                
                print(f"\n{Fore.GREEN}Thank you for using the Stock Predictor!{Style.RESET_ALL}")
                break
            
            # Add .NS suffix for Indian stocks if not present
            if not ticker.endswith('.NS') and not '.' in ticker:
                print(f"\nDo you want to analyze Indian stock? (y/n): ", end='')
                if input().lower().strip() == 'y':
                    ticker = f"{ticker}.NS"
            
            print(f"\n{Fore.YELLOW}Analyzing {ticker}... Please wait{Style.RESET_ALL}")
            
            # Get historical data with custom progress handling
            hist_data = predictor.download_data(ticker)
            
            if hist_data is None or len(hist_data) < 30:
                print(f"\n{Fore.RED}⚠ Insufficient historical data for {ticker}{Style.RESET_ALL}")
                continue
            
            # Show analysis progress
            print(f"{Fore.YELLOW}Processing technical analysis...{Style.RESET_ALL}")
            
            # Get next trading day
            next_trading_day = get_next_trading_day(ticker)
            
            # Make prediction with progress indication
            with tqdm(total=4, desc="Analyzing", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
                prediction = predictor.analyze_stock(ticker, hist_data)
                pbar.update(4)
            
            if prediction is None:
                print(f"\n{Fore.RED}⚠ Unable to analyze {ticker}{Style.RESET_ALL}")
                continue
            
            # Display results
            format_prediction_output(prediction, ticker, next_trading_day)
            
        except Exception as e:
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please try again with a different ticker.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

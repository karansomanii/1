import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import queue
from collections import defaultdict
import websocket
import json
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors
from ..advanced_analysis.integrated_analyzer import IntegratedAnalyzer

class RealTimeMonitor:
    def __init__(self):
        self.logger = Logger()
        self.analyzer = IntegratedAnalyzer()
        self.alert_queue = queue.Queue()
        self.active_stocks = set()
        self.stock_data = defaultdict(dict)
        self.alerts = defaultdict(list)
        self.thresholds = self._initialize_thresholds()
        self.ws = None
        self.is_running = False
        
    def _initialize_thresholds(self):
        """Initialize monitoring thresholds"""
        return {
            'price_movement': 0.02,  # 2% price movement
            'volume_spike': 3.0,     # 3x average volume
            'option_activity': 2.0,  # 2x average option volume
            'sentiment_change': 0.3,  # 30% sentiment change
            'liquidity_drop': 0.5,   # 50% liquidity drop
            'volatility_spike': 2.5  # 2.5x volatility increase
        }
    
    @handle_errors
    def start_monitoring(self, tickers):
        """Start real-time monitoring for given tickers"""
        try:
            self.active_stocks.update(tickers)
            self.is_running = True
            
            # Start monitoring threads
            self._start_websocket()
            self._start_alert_processor()
            self._start_analysis_thread()
            
            self.logger.info(f"Started monitoring {len(tickers)} stocks")
            
        except Exception as e:
            self.logger.error(f"Monitor start error: {e}")
            self.is_running = False
    
    def _start_websocket(self):
        """Initialize and start WebSocket connection"""
        try:
            websocket.enableTrace(True)
            self.ws = websocket.WebSocketApp(
                "wss://streamer.finance.yahoo.com",
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
        except Exception as e:
            self.logger.error(f"WebSocket start error: {e}")
    
    def _start_alert_processor(self):
        """Start alert processing thread"""
        try:
            alert_thread = threading.Thread(target=self._process_alerts)
            alert_thread.daemon = True
            alert_thread.start()
            
        except Exception as e:
            self.logger.error(f"Alert processor start error: {e}")
    
    def _start_analysis_thread(self):
        """Start continuous analysis thread"""
        try:
            analysis_thread = threading.Thread(target=self._continuous_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            
        except Exception as e:
            self.logger.error(f"Analysis thread start error: {e}")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'data' in data:
                self._process_market_data(data['data'])
                
        except Exception as e:
            self.logger.error(f"WebSocket message processing error: {e}")
    
    def _process_market_data(self, market_data):
        """Process incoming market data"""
        try:
            for ticker_data in market_data:
                ticker = ticker_data['s']
                if ticker in self.active_stocks:
                    # Update stock data
                    self.stock_data[ticker].update({
                        'price': ticker_data['p'],
                        'volume': ticker_data['v'],
                        'timestamp': datetime.now()
                    })
                    
                    # Check for alerts
                    self._check_alerts(ticker, ticker_data)
                    
        except Exception as e:
            self.logger.error(f"Market data processing error: {e}")
    
    def _check_alerts(self, ticker, data):
        """Check for alert conditions"""
        try:
            alerts = []
            
            # Price movement alerts
            if self._check_price_movement(ticker, data):
                alerts.append(self._create_alert(ticker, 'PRICE_MOVEMENT'))
            
            # Volume spike alerts
            if self._check_volume_spike(ticker, data):
                alerts.append(self._create_alert(ticker, 'VOLUME_SPIKE'))
            
            # Pattern completion alerts
            if self._check_pattern_completion(ticker):
                alerts.append(self._create_alert(ticker, 'PATTERN_COMPLETE'))
            
            # Options activity alerts
            if self._check_options_activity(ticker):
                alerts.append(self._create_alert(ticker, 'OPTIONS_ACTIVITY'))
            
            # Add alerts to queue
            for alert in alerts:
                self.alert_queue.put(alert)
                
        except Exception as e:
            self.logger.error(f"Alert check error: {e}")
    
    def _continuous_analysis(self):
        """Continuous analysis of monitored stocks"""
        try:
            while self.is_running:
                for ticker in self.active_stocks:
                    # Get latest data
                    data = self._get_analysis_data(ticker)
                    
                    # Run integrated analysis
                    analysis = self.analyzer.analyze_stock(ticker, data)
                    
                    if analysis:
                        # Update stock analysis
                        self.stock_data[ticker]['analysis'] = analysis
                        
                        # Check for analysis-based alerts
                        self._check_analysis_alerts(ticker, analysis)
                    
                # Sleep to prevent excessive CPU usage
                threading.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Continuous analysis error: {e}")
    
    def _process_alerts(self):
        """Process alerts from the queue"""
        try:
            while self.is_running:
                try:
                    # Get alert from queue with timeout
                    alert = self.alert_queue.get(timeout=1)
                    
                    # Process alert
                    self._handle_alert(alert)
                    
                    # Mark alert as processed
                    self.alert_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Alert processing error: {e}")
    
    def _handle_alert(self, alert):
        """Handle and distribute alerts"""
        try:
            # Log alert
            self.logger.info(f"Alert: {alert['type']} for {alert['ticker']}")
            
            # Store alert
            self.alerts[alert['ticker']].append(alert)
            
            # Trigger any alert-specific actions
            self._trigger_alert_actions(alert)
            
        except Exception as e:
            self.logger.error(f"Alert handling error: {e}")
    
    def _create_alert(self, ticker, alert_type):
        """Create alert object"""
        return {
            'ticker': ticker,
            'type': alert_type,
            'timestamp': datetime.now(),
            'data': self.stock_data[ticker].copy(),
            'priority': self._calculate_alert_priority(alert_type)
        }
    
    def get_active_alerts(self, ticker=None):
        """Get active alerts for a ticker or all tickers"""
        try:
            if ticker:
                return self.alerts[ticker]
            return dict(self.alerts)
            
        except Exception as e:
            self.logger.error(f"Alert retrieval error: {e}")
            return {}
    
    def stop_monitoring(self):
        """Stop monitoring and clean up"""
        try:
            self.is_running = False
            if self.ws:
                self.ws.close()
            self.active_stocks.clear()
            self.logger.info("Monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Monitor stop error: {e}") 
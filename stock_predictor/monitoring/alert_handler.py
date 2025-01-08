from datetime import datetime
import json
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors

class AlertHandler:
    def __init__(self):
        self.logger = Logger()
        self.alert_actions = {
            'PRICE_MOVEMENT': self._handle_price_alert,
            'VOLUME_SPIKE': self._handle_volume_alert,
            'PATTERN_COMPLETE': self._handle_pattern_alert,
            'OPTIONS_ACTIVITY': self._handle_options_alert,
            'ANALYSIS_SIGNAL': self._handle_analysis_alert
        }
    
    @handle_errors
    def handle_alert(self, alert):
        """Handle different types of alerts"""
        try:
            alert_type = alert['type']
            if alert_type in self.alert_actions:
                self.alert_actions[alert_type](alert)
            else:
                self.logger.warning(f"Unknown alert type: {alert_type}")
                
        except Exception as e:
            self.logger.error(f"Alert handling error: {e}")
    
    def _handle_price_alert(self, alert):
        """Handle price movement alerts"""
        try:
            message = (
                f"PRICE ALERT: {alert['ticker']}\n"
                f"Movement: {alert['data']['price_change']:.2%}\n"
                f"Current Price: {alert['data']['price']:.2f}\n"
                f"Volume: {alert['data']['volume']}\n"
                f"Time: {alert['timestamp']}"
            )
            self._send_alert(message, alert['priority'])
            
        except Exception as e:
            self.logger.error(f"Price alert handling error: {e}")
    
    def _handle_volume_alert(self, alert):
        """Handle volume spike alerts"""
        try:
            message = (
                f"VOLUME ALERT: {alert['ticker']}\n"
                f"Volume Spike: {alert['data']['volume_change']:.2f}x\n"
                f"Current Volume: {alert['data']['volume']}\n"
                f"Time: {alert['timestamp']}"
            )
            self._send_alert(message, alert['priority'])
            
        except Exception as e:
            self.logger.error(f"Volume alert handling error: {e}")
    
    def _send_alert(self, message, priority):
        """Send alert through configured channels"""
        try:
            # Log alert
            self.logger.info(message)
            
            # Add additional alert distribution channels here
            # (e.g., email, SMS, Telegram, etc.)
            
        except Exception as e:
            self.logger.error(f"Alert sending error: {e}") 
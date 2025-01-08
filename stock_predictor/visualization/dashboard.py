import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from ..monitoring.real_time_monitor import RealTimeMonitor
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors

class TradingDashboard:
    def __init__(self):
        self.logger = Logger()
        self.monitor = RealTimeMonitor()
        self.update_interval = 1  # seconds
        
    @handle_errors
    def run_dashboard(self):
        """Main dashboard function"""
        try:
            # Dashboard configuration
            st.set_page_config(
                page_title="Smart Trader Dashboard",
                page_icon="📈",
                layout="wide"
            )
            
            # Sidebar
            self._create_sidebar()
            
            # Main layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                self._display_main_charts()
            
            with col2:
                self._display_alerts_signals()
            
            # Bottom section
            self._display_portfolio_analysis()
            
            # Auto-refresh
            st.empty()
            
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
    
    def _create_sidebar(self):
        """Create dashboard sidebar"""
        try:
            st.sidebar.title("Smart Trader Controls")
            
            # Watchlist management
            st.sidebar.subheader("Watchlist")
            new_ticker = st.sidebar.text_input("Add Stock:")
            if st.sidebar.button("Add"):
                self.monitor.start_monitoring([new_ticker])
            
            # Display active stocks
            active_stocks = list(self.monitor.active_stocks)
            if active_stocks:
                selected_stock = st.sidebar.selectbox(
                    "Select Stock:",
                    active_stocks
                )
                if selected_stock:
                    st.session_state.selected_stock = selected_stock
            
            # Analysis settings
            st.sidebar.subheader("Analysis Settings")
            timeframe = st.sidebar.selectbox(
                "Timeframe:",
                ["1m", "5m", "15m", "1h", "1d"]
            )
            
            # Alert settings
            st.sidebar.subheader("Alert Settings")
            self._display_alert_settings()
            
        except Exception as e:
            self.logger.error(f"Sidebar creation error: {e}")
    
    def _display_main_charts(self):
        """Display main trading charts"""
        try:
            if 'selected_stock' not in st.session_state:
                st.warning("Please select a stock from the sidebar")
                return
            
            ticker = st.session_state.selected_stock
            data = self.monitor.stock_data.get(ticker, {})
            
            if not data:
                st.warning(f"No data available for {ticker}")
                return
            
            # Create main chart
            fig = self._create_main_chart(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            self._display_technical_indicators(data)
            
            # Order flow visualization
            self._display_order_flow(data)
            
        except Exception as e:
            self.logger.error(f"Main charts display error: {e}")
    
    def _create_main_chart(self, data):
        """Create main trading chart"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data['timestamp'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name="OHLC"
                ),
                row=1, col=1
            )
            
            # Volume bars
            fig.add_trace(
                go.Bar(
                    x=data['timestamp'],
                    y=data['volume'],
                    name="Volume"
                ),
                row=2, col=1
            )
            
            # Add technical indicators
            self._add_technical_overlays(fig, data)
            
            # Update layout
            fig.update_layout(
                title=f"{st.session_state.selected_stock} - Real-time Chart",
                xaxis_rangeslider_visible=False,
                height=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Main chart creation error: {e}")
            return None
    
    def _display_alerts_signals(self):
        """Display alerts and trading signals"""
        try:
            st.subheader("Alerts & Signals")
            
            # Active alerts
            alerts = self.monitor.get_active_alerts(st.session_state.selected_stock)
            if alerts:
                for alert in alerts:
                    self._display_alert_card(alert)
            
            # Trading signals
            st.subheader("Trading Signals")
            analysis = self.monitor.stock_data.get(
                st.session_state.selected_stock, {}
            ).get('analysis', {})
            
            if analysis:
                self._display_signal_card(analysis)
            
        except Exception as e:
            self.logger.error(f"Alerts and signals display error: {e}")
    
    def _display_portfolio_analysis(self):
        """Display portfolio analysis section"""
        try:
            st.header("Portfolio Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self._display_performance_metrics()
            
            with col2:
                self._display_risk_metrics()
            
            with col3:
                self._display_position_summary()
            
            # Portfolio composition chart
            fig = self._create_portfolio_chart()
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Portfolio analysis display error: {e}")
    
    def _display_alert_card(self, alert):
        """Display individual alert card"""
        try:
            with st.expander(f"{alert['type']} - {alert['timestamp'].strftime('%H:%M:%S')}"):
                st.write(f"Priority: {alert['priority']}")
                st.write(f"Details: {alert['data']}")
                
                if st.button("Dismiss", key=f"dismiss_{alert['timestamp']}"):
                    # Add alert dismissal logic
                    pass
                    
        except Exception as e:
            self.logger.error(f"Alert card display error: {e}")
    
    def _display_signal_card(self, analysis):
        """Display trading signal card"""
        try:
            recommendation = analysis.get('recommendation', {})
            if recommendation:
                color = {
                    'BUY': 'success',
                    'SELL': 'danger',
                    'HOLD': 'warning'
                }.get(recommendation['action'], 'primary')
                
                st.info(
                    f"**Action:** {recommendation['action']}\n\n"
                    f"**Confidence:** {recommendation['confidence']:.2%}\n\n"
                    f"**Entry Points:** {', '.join(map(str, recommendation['entry_points']))}\n\n"
                    f"**Stop Loss:** {recommendation['stop_loss']}\n\n"
                    f"**Take Profit:** {recommendation['take_profit']}"
                )
                
        except Exception as e:
            self.logger.error(f"Signal card display error: {e}")
    
    def _display_technical_indicators(self, data):
        """Display technical indicators"""
        try:
            st.subheader("Technical Indicators")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self._display_momentum_indicators(data)
            
            with col2:
                self._display_trend_indicators(data)
            
            with col3:
                self._display_volatility_indicators(data)
                
        except Exception as e:
            self.logger.error(f"Technical indicators display error: {e}")
    
    def _display_order_flow(self, data):
        """Display order flow visualization"""
        try:
            st.subheader("Order Flow Analysis")
            
            # Create order flow heat map
            fig = go.Figure(data=[
                go.Heatmap(
                    z=data.get('order_flow_matrix', []),
                    x=data.get('price_levels', []),
                    y=data.get('time_levels', []),
                    colorscale='RdYlGn'
                )
            ])
            
            fig.update_layout(
                title="Order Flow Heat Map",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Order flow display error: {e}") 
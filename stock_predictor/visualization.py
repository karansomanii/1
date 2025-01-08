import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    def __init__(self):
        # Set style for matplotlib
        plt.style.use('seaborn')
        sns.set_theme(style="darkgrid")
        
    def plot_prediction_results(self, data, prediction, save_path=None):
        """Create interactive plot of prediction results"""
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price Action', 'Volume & Indicators'),
                row_heights=[0.7, 0.3]
            )

            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )

            # Add prediction lines
            current_price = prediction['current_price']
            target_price = prediction['target_price']
            stop_loss = prediction['stop_loss']

            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="white",
                annotation_text="Current",
                row=1, col=1
            )
            fig.add_hline(
                y=target_price,
                line_dash="dash",
                line_color="green",
                annotation_text="Target",
                row=1, col=1
            )
            fig.add_hline(
                y=stop_loss,
                line_dash="dash",
                line_color="red",
                annotation_text="Stop Loss",
                row=1, col=1
            )

            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(100,100,100,0.5)'
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                title=f"Prediction Analysis for {prediction['ticker']}",
                yaxis_title="Price",
                yaxis2_title="Volume",
                xaxis_rangeslider_visible=False,
                height=800
            )

            if save_path:
                fig.write_html(save_path)
            
            return fig

        except Exception as e:
            print(f"Error creating prediction plot: {e}")
            return None

    def plot_technical_analysis(self, data, analysis, save_path=None):
        """Plot technical analysis indicators"""
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(3, 2)

            # Price and MA plot
            ax1 = fig.add_subplot(gs[0, :])
            data['Close'].plot(ax=ax1, label='Close Price')
            if 'trends' in analysis:
                for ma in analysis['trends']:
                    if ma.startswith('MA'):
                        ax1.plot(data.index, [analysis['trends'][ma]['value']] * len(data),
                                label=ma, linestyle='--')
            ax1.set_title('Price and Moving Averages')
            ax1.legend()

            # RSI
            ax2 = fig.add_subplot(gs[1, 0])
            if 'indicators' in analysis and 'RSI' in analysis['indicators']:
                rsi_data = pd.Series(analysis['indicators']['RSI'], index=data.index)
                rsi_data.plot(ax=ax2)
                ax2.axhline(70, color='r', linestyle='--')
                ax2.axhline(30, color='g', linestyle='--')
            ax2.set_title('RSI')

            # MACD
            ax3 = fig.add_subplot(gs[1, 1])
            if 'indicators' in analysis and 'MACD' in analysis['indicators']:
                macd_data = pd.Series(analysis['indicators']['MACD']['histogram'], index=data.index)
                macd_data.plot(ax=ax3, kind='bar')
            ax3.set_title('MACD Histogram')

            # Volatility
            ax4 = fig.add_subplot(gs[2, 0])
            if 'volatility' in analysis:
                vol_data = pd.Series(analysis['volatility']['daily_volatility'], index=data.index)
                vol_data.plot(ax=ax4)
            ax4.set_title('Volatility')

            # Volume
            ax5 = fig.add_subplot(gs[2, 1])
            data['Volume'].plot(ax=ax5, kind='bar')
            ax5.set_title('Volume')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                plt.close()
            
            return fig

        except Exception as e:
            print(f"Error creating technical analysis plot: {e}")
            return None

    def plot_performance_metrics(self, performance_data, save_path=None):
        """Plot performance metrics"""
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)

            # Equity curve
            ax1 = fig.add_subplot(gs[0, :])
            equity_curve = pd.Series(performance_data['equity_curve'])
            equity_curve.plot(ax=ax1)
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value')

            # Win/Loss ratio
            ax2 = fig.add_subplot(gs[1, 0])
            wins = performance_data['successful_predictions']
            losses = performance_data['total_predictions'] - wins
            ax2.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%')
            ax2.set_title('Win/Loss Ratio')

            # Returns distribution
            ax3 = fig.add_subplot(gs[1, 1])
            returns = pd.Series(performance_data.get('returns', []))
            if not returns.empty:
                returns.hist(ax=ax3, bins=50)
                ax3.axvline(returns.mean(), color='r', linestyle='--')
            ax3.set_title('Returns Distribution')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                plt.close()
            
            return fig

        except Exception as e:
            print(f"Error creating performance plot: {e}")
            return None

    def create_risk_dashboard(self, risk_analysis, save_path=None):
        """Create risk analysis dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Risk Metrics',
                    'Value at Risk',
                    'Market Risk Exposure',
                    'Position Risk'
                )
            )

            # Risk metrics gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_analysis['risk_score'],
                    title={'text': "Risk Score"},
                    gauge={'axis': {'range': [0, 100]},
                           'steps': [
                               {'range': [0, 20], 'color': "green"},
                               {'range': [20, 40], 'color': "lightgreen"},
                               {'range': [40, 60], 'color': "yellow"},
                               {'range': [60, 80], 'color': "orange"},
                               {'range': [80, 100], 'color': "red"}
                           ]},
                ),
                row=1, col=1
            )

            # VaR plot
            var_values = [
                risk_analysis['value_at_risk']['daily_var_95'],
                risk_analysis['value_at_risk']['daily_var_99']
            ]
            fig.add_trace(
                go.Bar(
                    x=['95% VaR', '99% VaR'],
                    y=var_values,
                    name='Value at Risk'
                ),
                row=1, col=2
            )

            # Market risk
            fig.add_trace(
                go.Scatter(
                    x=['Beta', 'Systematic Risk', 'Required Return'],
                    y=[
                        risk_analysis['market_risk']['beta'],
                        risk_analysis['market_risk']['systematic_risk'],
                        risk_analysis['market_risk']['required_return']
                    ],
                    mode='lines+markers',
                    name='Market Risk'
                ),
                row=2, col=1
            )

            # Position risk
            fig.add_trace(
                go.Bar(
                    x=['Potential Gain', 'Potential Loss', 'Risk-Reward'],
                    y=[
                        risk_analysis['position_risk']['potential_gain'],
                        risk_analysis['position_risk']['potential_loss'],
                        risk_analysis['position_risk']['risk_reward_ratio']
                    ],
                    name='Position Risk'
                ),
                row=2, col=2
            )

            fig.update_layout(
                height=800,
                showlegend=False,
                title_text="Risk Analysis Dashboard"
            )

            if save_path:
                fig.write_html(save_path)
            
            return fig

        except Exception as e:
            print(f"Error creating risk dashboard: {e}")
            return None 
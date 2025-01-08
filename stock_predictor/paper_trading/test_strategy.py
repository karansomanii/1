from datetime import datetime, timedelta
import yfinance as yf
from .paper_trading_system import PaperTradingSystem
from ..advanced_analysis.integrated_analyzer import IntegratedAnalyzer
from ..utils.logger import Logger

class StrategyTester:
    def __init__(self, initial_capital=100000):
        self.paper_trader = PaperTradingSystem(initial_capital)
        self.analyzer = IntegratedAnalyzer()
        self.logger = Logger()
        
    def run_test(self, tickers, days=30):
        """Run paper trading test"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
            
            for ticker in tickers:
                # Get historical data
                data = yf.download(ticker, start=start_date, end=end_date)
                
                for index, row in data.iterrows():
                    # Update market data
                    self.paper_trader.update_market_data(
                        ticker=ticker,
                        current_price=row['Close'],
                        timestamp=index
                    )
                    
                    # Run analysis
                    analysis = self.analyzer.analyze_stock(ticker, data.loc[:index])
                    
                    # Generate and execute signals
                    if analysis and analysis.get('recommendation'):
                        self._execute_signals(ticker, analysis['recommendation'], row['Close'])
                    
                    # Print daily summary
                    if index.time() == data.index[-1].time():
                        self._print_daily_summary()
            
            # Print final results
            self._print_final_results()
            
        except Exception as e:
            self.logger.error(f"Strategy test error: {e}")
    
    def _execute_signals(self, ticker, recommendation, current_price):
        """Execute trading signals"""
        try:
            action = recommendation['action']
            confidence = recommendation['confidence']
            
            if action == 'BUY' and confidence > 0.7:
                # Calculate position size
                capital = self.paper_trader.current_capital
                position_size = capital * 0.02  # 2% of capital per trade
                quantity = int(position_size / current_price)
                
                if quantity > 0:
                    self.paper_trader.place_order(
                        ticker=ticker,
                        order_type='BUY',
                        quantity=quantity,
                        price=current_price,
                        stop_loss=recommendation['stop_loss'],
                        target=recommendation['take_profit']
                    )
            
            elif action == 'SELL':
                # Close position if exists
                if ticker in self.paper_trader.positions:
                    position = self.paper_trader.positions[ticker]
                    self.paper_trader.place_order(
                        ticker=ticker,
                        order_type='SELL',
                        quantity=position['quantity'],
                        price=current_price
                    )
                    
        except Exception as e:
            self.logger.error(f"Signal execution error: {e}")
    
    def _print_daily_summary(self):
        """Print daily portfolio summary"""
        summary = self.paper_trader.get_portfolio_summary()
        print("\nDaily Summary:")
        print(f"Total Value: {summary['total_value']:.2f}")
        print(f"Cash: {summary['cash']:.2f}")
        print(f"Daily P&L: {summary['daily_pnl']:.2f}")
        print("\nPositions:")
        for pos in summary['positions']:
            print(f"{pos['ticker']}: {pos['quantity']} shares, P&L: {pos['unrealized_pnl']:.2f}")
    
    def _print_final_results(self):
        """Print final testing results"""
        summary = self.paper_trader.get_portfolio_summary()
        print("\n=== Final Results ===")
        print(f"Initial Capital: {self.paper_trader.initial_capital:.2f}")
        print(f"Final Value: {summary['total_value']:.2f}")
        print(f"Total Return: {((summary['total_value'] / self.paper_trader.initial_capital) - 1) * 100:.2f}%")
        print(f"Number of Trades: {len(self.paper_trader.trades_history)}")
        
        if self.paper_trader.trades_history:
            winning_trades = len([t for t in self.paper_trader.trades_history if t['realized_pnl'] > 0])
            win_rate = winning_trades / len(self.paper_trader.trades_history) * 100
            print(f"Win Rate: {win_rate:.2f}%")
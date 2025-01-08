import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from ..utils.logger import Logger
from ..utils.error_handler import handle_errors

class PaperTradingSystem:
    def __init__(self, initial_capital=100000):
        self.logger = Logger()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades_history = []
        self.pending_orders = {}
        self.filled_orders = {}
        self.daily_stats = defaultdict(dict)
        
    @handle_errors
    def place_order(self, ticker, order_type, quantity, price=None, stop_loss=None, target=None):
        """Place a paper trading order"""
        try:
            order_id = f"ORDER_{len(self.pending_orders) + len(self.filled_orders) + 1}"
            timestamp = datetime.now()
            
            order = {
                'order_id': order_id,
                'ticker': ticker,
                'type': order_type,
                'quantity': quantity,
                'price': price,
                'stop_loss': stop_loss,
                'target': target,
                'status': 'PENDING',
                'timestamp': timestamp
            }
            
            # Validate order
            if self._validate_order(order):
                self.pending_orders[order_id] = order
                self.logger.info(f"Order placed: {order_id} for {ticker}")
                return order_id
            else:
                self.logger.warning(f"Order validation failed for {ticker}")
                return None
                
        except Exception as e:
            self.logger.error(f"Order placement error: {e}")
            return None
    
    def _validate_order(self, order):
        """Validate if order can be placed"""
        try:
            if order['type'] == 'BUY':
                required_capital = order['quantity'] * order['price']
                if required_capital > self.current_capital:
                    self.logger.warning("Insufficient capital for order")
                    return False
            
            elif order['type'] == 'SELL':
                if order['ticker'] not in self.positions:
                    self.logger.warning(f"No position found for {order['ticker']}")
                    return False
                if self.positions[order['ticker']]['quantity'] < order['quantity']:
                    self.logger.warning("Insufficient quantity for sell order")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Order validation error: {e}")
            return False
    
    @handle_errors
    def update_market_data(self, ticker, current_price, timestamp=None):
        """Update market data and process pending orders"""
        try:
            timestamp = timestamp or datetime.now()
            
            # Process pending orders
            self._process_pending_orders(ticker, current_price, timestamp)
            
            # Update positions
            if ticker in self.positions:
                position = self.positions[ticker]
                old_value = position['market_value']
                position['market_value'] = position['quantity'] * current_price
                position['unrealized_pnl'] = position['market_value'] - position['cost_basis']
                position['last_price'] = current_price
                position['last_update'] = timestamp
                
                # Check stop loss and target
                self._check_exit_conditions(ticker, current_price, timestamp)
                
            # Update daily stats
            self._update_daily_stats(ticker, current_price, timestamp)
            
        except Exception as e:
            self.logger.error(f"Market data update error: {e}")
    
    def _process_pending_orders(self, ticker, current_price, timestamp):
        """Process pending orders based on current price"""
        try:
            orders_to_remove = []
            
            for order_id, order in self.pending_orders.items():
                if order['ticker'] != ticker:
                    continue
                    
                if self._should_fill_order(order, current_price):
                    self._fill_order(order_id, current_price, timestamp)
                    orders_to_remove.append(order_id)
            
            # Remove filled orders from pending
            for order_id in orders_to_remove:
                del self.pending_orders[order_id]
                
        except Exception as e:
            self.logger.error(f"Order processing error: {e}")
    
    def _fill_order(self, order_id, fill_price, timestamp):
        """Fill a pending order"""
        try:
            order = self.pending_orders[order_id]
            order['fill_price'] = fill_price
            order['fill_timestamp'] = timestamp
            order['status'] = 'FILLED'
            
            # Update positions
            ticker = order['ticker']
            quantity = order['quantity']
            
            if order['type'] == 'BUY':
                cost = quantity * fill_price
                self.current_capital -= cost
                
                if ticker not in self.positions:
                    self.positions[ticker] = {
                        'quantity': quantity,
                        'cost_basis': cost,
                        'market_value': cost,
                        'unrealized_pnl': 0,
                        'last_price': fill_price,
                        'last_update': timestamp
                    }
                else:
                    pos = self.positions[ticker]
                    new_quantity = pos['quantity'] + quantity
                    new_cost = pos['cost_basis'] + cost
                    pos['quantity'] = new_quantity
                    pos['cost_basis'] = new_cost
                    pos['market_value'] = new_quantity * fill_price
            
            elif order['type'] == 'SELL':
                revenue = quantity * fill_price
                self.current_capital += revenue
                
                pos = self.positions[ticker]
                pos['quantity'] -= quantity
                realized_pnl = revenue - (pos['cost_basis'] / pos['quantity']) * quantity
                
                if pos['quantity'] == 0:
                    del self.positions[ticker]
                else:
                    pos['cost_basis'] -= (pos['cost_basis'] / pos['quantity']) * quantity
                    pos['market_value'] = pos['quantity'] * fill_price
                
                # Record realized PnL
                self.trades_history.append({
                    'order_id': order_id,
                    'ticker': ticker,
                    'quantity': quantity,
                    'entry_price': pos['cost_basis'] / pos['quantity'],
                    'exit_price': fill_price,
                    'realized_pnl': realized_pnl,
                    'timestamp': timestamp
                })
            
            self.filled_orders[order_id] = order
            self.logger.info(f"Order filled: {order_id} at {fill_price}")
            
        except Exception as e:
            self.logger.error(f"Order fill error: {e}")
    
    def get_portfolio_summary(self):
        """Get current portfolio summary"""
        try:
            total_value = self.current_capital
            positions_summary = []
            
            for ticker, position in self.positions.items():
                total_value += position['market_value']
                positions_summary.append({
                    'ticker': ticker,
                    'quantity': position['quantity'],
                    'cost_basis': position['cost_basis'],
                    'market_value': position['market_value'],
                    'unrealized_pnl': position['unrealized_pnl'],
                    'last_price': position['last_price']
                })
            
            return {
                'total_value': total_value,
                'cash': self.current_capital,
                'positions': positions_summary,
                'daily_pnl': self._calculate_daily_pnl(),
                'total_pnl': total_value - self.initial_capital
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio summary error: {e}")
            return None 
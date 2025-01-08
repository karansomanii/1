import argparse
from .paper_trading.test_strategy import StrategyTester
from .utils.logger import Logger

def main():
    logger = Logger()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Predictor Trading System')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital for paper trading')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to test')
    parser.add_argument('--stocks', nargs='+', 
                       default=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
                       help='List of stock symbols to trade')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run the strategy tester
        tester = StrategyTester(initial_capital=args.capital)
        tester.run_test(args.stocks, days=args.days)
        
    except Exception as e:
        logger.error(f"Error running strategy: {e}")
        raise

if __name__ == '__main__':
    main() 
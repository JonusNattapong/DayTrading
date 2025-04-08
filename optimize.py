"""
Command-line interface for optimizing day trading strategies.
This script provides a simple way to optimize strategy parameters.
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from data.data_loader import download_data, load_data, process_data_for_day_trading
from utils.optimizer import StrategyOptimizer, optimize_strategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)


def load_data_for_optimization(config, symbol, timeframe):
    """Load data for optimization."""
    # Get data
    data_dir = config['data'].get('data_dir', 'data')
    use_cached = config['data'].get('use_cached_data', True)
    start_date = config['backtest']['start_date']
    end_date = config['backtest']['end_date']
    
    # Create potential filename for cached data
    if isinstance(start_date, str):
        start_str = start_date.replace('-', '')
    else:
        start_str = start_date.strftime('%Y%m%d')
    
    if isinstance(end_date, str):
        end_str = end_date.replace('-', '')
    else:
        end_str = end_date.strftime('%Y%m%d')
        
    cache_filename = f"{symbol}_{timeframe}_{start_str}_{end_str}.csv"
    cache_path = os.path.join(data_dir, cache_filename)
    
    # Load or download data
    if use_cached and os.path.exists(cache_path):
        df = load_data(cache_path)
        logger.info(f"Loaded cached data from {cache_path}")
    else:
        df = download_data(symbol, start_date, end_date, timeframe, True, data_dir)
        logger.info(f"Downloaded data for {symbol}")
    
    if df is None or df.empty:
        logger.error(f"No data available for {symbol} on {timeframe} timeframe")
        sys.exit(1)
    
    # Process data for day trading if trading hours are specified
    if 'trading_hours' in config['backtest']:
        start_hour = config['backtest']['trading_hours']['start']
        end_hour = config['backtest']['trading_hours']['end']
        df = process_data_for_day_trading(df, (start_hour, end_hour))
        
        if df is None or df.empty:
            logger.error(f"No data available after filtering for trading hours")
            sys.exit(1)
    
    return df


def define_default_param_grid():
    """Define a default parameter grid for optimization."""
    return {
        'rsi_period': [7, 14, 21],
        'rsi_oversold': [20, 30, 40],
        'rsi_overbought': [60, 70, 80],
        'sma_fast_period': [10, 20, 50],
        'sma_slow_period': [50, 100, 200],
        'bb_period': [10, 20, 30],
        'bb_std': [1.5, 2.0, 2.5],
        'stop_loss_pct': [0.5, 1.0, 2.0],
        'take_profit_pct': [1.0, 2.0, 3.0]
    }


def define_scoring_function(scoring_name):
    """Define a scoring function based on name."""
    if scoring_name == 'total_return':
        return lambda perf: perf.get('total_return', 0)
    elif scoring_name == 'sharpe':
        return lambda perf: perf.get('total_return', 0) / max(0.1, abs(perf.get('max_drawdown', 1)))
    elif scoring_name == 'profit_factor':
        return lambda perf: perf.get('profit_factor', 0)
    elif scoring_name == 'win_rate':
        return lambda perf: perf.get('win_rate', 0)
    elif scoring_name == 'combined':
        # Weighted combination of multiple metrics
        return lambda perf: (
            0.4 * perf.get('total_return', 0) - 
            0.3 * abs(perf.get('max_drawdown', 0)) + 
            0.3 * perf.get('profit_factor', 0) * 10
        )
    else:
        return scoring_name  # Use the name as a key in the performance dict


def main():
    """Main function for the optimization script."""
    parser = argparse.ArgumentParser(description="Optimize day trading strategy parameters")
    parser.add_argument("--config", default="config/config.json", help="Path to configuration file")
    parser.add_argument("--symbol", help="Symbol to optimize for (overrides config)")
    parser.add_argument("--timeframe", help="Timeframe to optimize for (overrides config)")
    parser.add_argument("--max-combinations", type=int, default=100, help="Maximum number of parameter combinations to test")
    parser.add_argument("--scoring", default="combined", choices=["total_return", "sharpe", "profit_factor", "win_rate", "combined"], help="Scoring function to use")
    parser.add_argument("--output", default=None, help="Path to save optimization results")
    parser.add_argument("--report", default=None, help="Path to save optimization report")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get symbol and timeframe
    symbol = args.symbol or config['backtest']['symbols'][0]
    timeframe = args.timeframe or config['strategy']['timeframes'][0]
    
    logger.info(f"Optimizing strategy for {symbol} on {timeframe} timeframe")
    
    # Load data
    df = load_data_for_optimization(config, symbol, timeframe)
    
    # Define parameter grid
    param_grid = define_default_param_grid()
    
    # Define scoring function
    scoring = define_scoring_function(args.scoring)
    
    # Run optimization
    logger.info("Starting optimization...")
    best_params, best_score, optimizer = optimize_strategy(
        df, 
        param_grid, 
        timeframe=timeframe, 
        initial_capital=config['backtest']['initial_capital'],
        scoring=scoring,
        max_combinations=args.max_combinations
    )
    
    # Print results
    print("\n===== OPTIMIZATION RESULTS =====")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Best score: {best_score}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Get performance of best strategy
    top_strategies = optimizer.get_top_n_strategies(1)
    if top_strategies:
        best_performance = top_strategies[0]['performance']
        print("\nPerformance:")
        print(f"  Total trades: {best_performance.get('total_trades', 0)}")
        print(f"  Win rate: {best_performance.get('win_rate', 0) * 100:.2f}%")
        print(f"  Total return: {best_performance.get('total_return', 0):.2f}%")
        print(f"  Max drawdown: {best_performance.get('max_drawdown', 0):.2f}%")
        print(f"  Profit factor: {best_performance.get('profit_factor', 0):.2f}")
    
    # Show top 5 strategies
    print("\nTop 5 strategies:")
    top_strategies = optimizer.get_top_n_strategies(5)
    for i, strategy in enumerate(top_strategies):
        print(f"\n#{i+1} (Score: {strategy['score']:.2f}):")
        print("  Parameters:")
        for param, value in strategy['params'].items():
            print(f"    {param}: {value}")
        print("  Performance:")
        perf = strategy['performance']
        print(f"    Win rate: {perf.get('win_rate', 0) * 100:.2f}%")
        print(f"    Total return: {perf.get('total_return', 0):.2f}%")
        print(f"    Max drawdown: {perf.get('max_drawdown', 0):.2f}%")
    
    # Save results if requested
    if args.output:
        output_path = args.output
        optimizer.save_results(output_path)
        print(f"\nOptimization results saved to: {output_path}")
    
    # Generate report if requested
    if args.report:
        report_path = args.report
        optimizer.generate_report(report_path)
        print(f"\nOptimization report generated: {report_path}")
    else:
        # Use default path
        results_dir = config['output'].get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(results_dir, f"optimization_report_{symbol}_{timeframe}_{timestamp}.html")
        optimizer.generate_report(report_path)
        print(f"\nOptimization report generated: {report_path}")


if __name__ == "__main__":
    main()
"""
Main module for the Day Trading Strategy application.
Serves as the entry point to run backtests and analyze results.
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from data.data_loader import download_data, load_data, process_data_for_day_trading
from strategy.day_trading_strategy import DayTradingStrategy
from backtest.backtester import DayTradingBacktester
from visualization.chart_visualizer import (
    plot_price_with_indicators, 
    plot_trades, 
    plot_equity_curve, 
    plot_performance_summary
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("day_trading.log"),
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


def ensure_directories(config):
    """Ensure required directories exist."""
    directories = [
        config['data'].get('data_dir', 'data'),
        config['output'].get('results_dir', 'results')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def run_backtest(config):
    """Run backtest based on configuration."""
    # Extract configuration
    symbols = config['backtest']['symbols']
    start_date = config['backtest']['start_date']
    end_date = config['backtest']['end_date']
    initial_capital = config['backtest']['initial_capital']
    timeframes = config['strategy']['timeframes']
    
    # Create results directory
    results_dir = config['output'].get('results_dir', 'results')
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{current_time}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create strategy
    strategy_config = {
        **config['strategy']['indicators'],
        **config['strategy']['risk_management']
    }
    strategy = DayTradingStrategy(strategy_config)
    
    # Run backtest for each symbol and timeframe
    overall_results = {}
    
    for symbol in symbols:
        symbol_results = {}
        
        for timeframe in timeframes:
            logger.info(f"Running backtest for {symbol} on {timeframe} timeframe")
            
            # Get data
            data_dir = config['data'].get('data_dir', 'data')
            use_cached = config['data'].get('use_cached_data', True)
            
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
                logger.warning(f"No data available for {symbol} on {timeframe} timeframe")
                continue
            
            # Process data for day trading if trading hours are specified
            if 'trading_hours' in config['backtest']:
                start_hour = config['backtest']['trading_hours']['start']
                end_hour = config['backtest']['trading_hours']['end']
                df = process_data_for_day_trading(df, (start_hour, end_hour))
                
                if df is None or df.empty:
                    logger.warning(f"No data available after filtering for trading hours")
                    continue
            
            # Run backtest
            backtester = DayTradingBacktester(strategy, initial_capital)
            results_df = backtester.run(df, timeframe)
            
            # Get performance summary
            performance = backtester.get_performance_summary()
            
            # Save results
            if config['output'].get('save_results', True):
                # Save performance summary
                performance_path = os.path.join(run_dir, f"{symbol}_{timeframe}_performance.json")
                with open(performance_path, 'w') as f:
                    json.dump(performance, f, indent=4)
                
                # Save trades
                if config['output'].get('save_trades', True):
                    trades_df = results_df[results_df['pnl'] != 0].copy()
                    trades_path = os.path.join(run_dir, f"{symbol}_{timeframe}_trades.csv")
                    trades_df.to_csv(trades_path)
                
                # Generate plots
                if config['output'].get('plot_charts', True):
                    # Plot price with indicators
                    chart_path = os.path.join(run_dir, f"{symbol}_{timeframe}_chart.png")
                    plot_price_with_indicators(results_df, chart_path, f"{symbol} - {timeframe} Chart with Indicators")
                    
                    # Plot trades
                    trades_path = os.path.join(run_dir, f"{symbol}_{timeframe}_trades_chart.png")
                    plot_trades(results_df, trades_path, f"{symbol} - {timeframe} Trades")
                    
                    # Plot equity curve
                    equity_path = os.path.join(run_dir, f"{symbol}_{timeframe}_equity.png")
                    plot_equity_curve(results_df, equity_path, f"{symbol} - {timeframe} Equity Curve")
                    
                    # Plot performance summary
                    perf_chart_path = os.path.join(run_dir, f"{symbol}_{timeframe}_performance.png")
                    plot_performance_summary(performance, perf_chart_path)
            
            symbol_results[timeframe] = performance
            logger.info(f"Backtest completed for {symbol} on {timeframe} timeframe")
        
        overall_results[symbol] = symbol_results
    
    # Save overall results
    overall_path = os.path.join(run_dir, "overall_results.json")
    with open(overall_path, 'w') as f:
        json.dump(overall_results, f, indent=4)
    
    return overall_results, run_dir


def main():
    """Main entry point of the application."""
    parser = argparse.ArgumentParser(description="Day Trading Strategy Backtester")
    parser.add_argument("--config", default="config/config.json", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Ensure directories exist
    ensure_directories(config)
    
    # Run backtest
    logger.info("Starting backtest")
    results, results_dir = run_backtest(config)
    logger.info(f"Backtest completed. Results saved to {results_dir}")
    
    # Print summary
    print("\n===== BACKTEST RESULTS SUMMARY =====")
    for symbol, timeframe_results in results.items():
        print(f"\n{symbol}:")
        for timeframe, performance in timeframe_results.items():
            print(f"  {timeframe}:")
            print(f"    Total Trades: {performance.get('total_trades', 0)}")
            print(f"    Win Rate: {performance.get('win_rate', 0) * 100:.2f}%")
            print(f"    Total Return: {performance.get('total_return', 0):.2f}%")
            print(f"    Max Drawdown: {performance.get('max_drawdown', 0):.2f}%")
    
    print(f"\nDetailed results saved to: {results_dir}")


if __name__ == "__main__":
    main()
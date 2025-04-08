"""
Utility module for analyzing trading results.
Provides functions to compare strategies, analyze performance, and generate reports.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def load_results(results_dir):
    """
    Load results from a backtest run directory.
    
    Parameters:
    -----------
    results_dir : str
        Path to the directory containing backtest results
        
    Returns:
    --------
    dict
        Dictionary with performance metrics for all symbols and timeframes
    """
    try:
        # Load overall results
        overall_path = os.path.join(results_dir, "overall_results.json")
        if not os.path.exists(overall_path):
            logger.warning(f"Overall results file not found: {overall_path}")
            return None
        
        with open(overall_path, 'r') as f:
            results = json.load(f)
        
        return results
    
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return None


def load_trade_data(results_dir, symbol, timeframe):
    """
    Load trade data from a CSV file.
    
    Parameters:
    -----------
    results_dir : str
        Path to the directory containing backtest results
    symbol : str
        Symbol to load trades for
    timeframe : str
        Timeframe to load trades for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with trade data
    """
    try:
        # Build path to trades CSV
        trades_path = os.path.join(results_dir, f"{symbol}_{timeframe}_trades.csv")
        if not os.path.exists(trades_path):
            logger.warning(f"Trades file not found: {trades_path}")
            return None
        
        # Load trades
        trades_df = pd.read_csv(trades_path, index_col=0, parse_dates=True)
        return trades_df
    
    except Exception as e:
        logger.error(f"Error loading trade data: {str(e)}")
        return None


def analyze_trades(trades_df):
    """
    Analyze trades and generate trade statistics.
    
    Parameters:
    -----------
    trades_df : pandas.DataFrame
        DataFrame with trade data
        
    Returns:
    --------
    dict
        Dictionary with trade analysis metrics
    """
    if trades_df is None or trades_df.empty:
        return {"error": "No trade data available"}
    
    # Calculate basic statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_trade_pnl": 0,
            "std_dev_pnl": 0
        }
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    
    total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate advanced statistics
    avg_trade_pnl = trades_df['pnl'].mean()
    std_dev_pnl = trades_df['pnl'].std()
    
    # Calculate trading edge (expectancy)
    edge = win_rate * avg_win - (1 - win_rate) * abs(avg_loss) if avg_loss < 0 else 0
    
    # Calculate consecutive wins/losses
    trades_df['win'] = trades_df['pnl'] > 0
    consecutive_wins = []
    consecutive_losses = []
    current_streak = 1
    
    for i in range(1, len(trades_df)):
        if trades_df.iloc[i]['win'] == trades_df.iloc[i-1]['win']:
            current_streak += 1
        else:
            if trades_df.iloc[i-1]['win']:
                consecutive_wins.append(current_streak)
            else:
                consecutive_losses.append(current_streak)
            current_streak = 1
    
    # Add the last streak
    if len(trades_df) > 0:
        if trades_df.iloc[-1]['win']:
            consecutive_wins.append(current_streak)
        else:
            consecutive_losses.append(current_streak)
    
    max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
    max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_trade_pnl": avg_trade_pnl,
        "std_dev_pnl": std_dev_pnl,
        "trading_edge": edge,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses
    }


def compare_timeframes(results):
    """
    Compare performance across different timeframes.
    
    Parameters:
    -----------
    results : dict
        Dictionary with performance metrics
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with performance comparison
    """
    if not results:
        return pd.DataFrame()
    
    comparison_data = []
    
    for symbol, timeframe_results in results.items():
        for timeframe, performance in timeframe_results.items():
            row = {
                'Symbol': symbol,
                'Timeframe': timeframe,
                'Total Trades': performance.get('total_trades', 0),
                'Win Rate': performance.get('win_rate', 0) * 100,
                'Total Return (%)': performance.get('total_return', 0),
                'Max Drawdown (%)': performance.get('max_drawdown', 0),
                'Profit Factor': performance.get('profit_factor', 0)
            }
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    return df


def generate_performance_report(results_dir, output_file=None):
    """
    Generate a comprehensive performance report.
    
    Parameters:
    -----------
    results_dir : str
        Path to the directory containing backtest results
    output_file : str
        Path to the output HTML file
        
    Returns:
    --------
    str
        Path to the generated report
    """
    results = load_results(results_dir)
    
    if not results:
        logger.error("Could not load results for report generation")
        return None
    
    # Create comparison table
    comparison_df = compare_timeframes(results)
    
    # Create HTML report
    html_content = []
    html_content.append("<html><head>")
    html_content.append("<style>")
    html_content.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html_content.append("h1, h2, h3 { color: #2c3e50; }")
    html_content.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
    html_content.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html_content.append("th { background-color: #f2f2f2; }")
    html_content.append("tr:nth-child(even) { background-color: #f9f9f9; }")
    html_content.append(".good { color: green; }")
    html_content.append(".bad { color: red; }")
    html_content.append("</style>")
    html_content.append("<title>Day Trading Strategy Performance Report</title>")
    html_content.append("</head><body>")
    
    # Add header
    html_content.append("<h1>Day Trading Strategy Performance Report</h1>")
    html_content.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    
    # Add comparison table
    html_content.append("<h2>Strategy Performance Comparison</h2>")
    html_content.append(comparison_df.to_html(index=False))
    
    # Add details for each symbol and timeframe
    html_content.append("<h2>Detailed Performance</h2>")
    
    for symbol, timeframe_results in results.items():
        html_content.append(f"<h3>{symbol}</h3>")
        
        for timeframe, performance in timeframe_results.items():
            html_content.append(f"<h4>Timeframe: {timeframe}</h4>")
            
            # Create performance table
            html_content.append("<table>")
            html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
            
            # Add rows
            metrics = [
                ("Total Trades", performance.get('total_trades', 0)),
                ("Winning Trades", performance.get('winning_trades', 0)),
                ("Losing Trades", performance.get('losing_trades', 0)),
                ("Win Rate", f"{performance.get('win_rate', 0) * 100:.2f}%"),
                ("Average Win", f"{performance.get('avg_win', 0):.2f}%"),
                ("Average Loss", f"{performance.get('avg_loss', 0):.2f}%"),
                ("Profit Factor", f"{performance.get('profit_factor', 0):.2f}"),
                ("Total Return", f"{performance.get('total_return', 0):.2f}%"),
                ("Max Drawdown", f"{performance.get('max_drawdown', 0):.2f}%")
            ]
            
            for metric, value in metrics:
                html_content.append(f"<tr><td>{metric}</td><td>{value}</td></tr>")
            
            html_content.append("</table>")
            
            # Add images
            chart_path = f"{symbol}_{timeframe}_chart.png"
            if os.path.exists(os.path.join(results_dir, chart_path)):
                html_content.append(f"<h4>Price Chart with Indicators</h4>")
                html_content.append(f"<img src='{chart_path}' style='width:100%;max-width:800px;'/>")
            
            trades_chart_path = f"{symbol}_{timeframe}_trades_chart.png"
            if os.path.exists(os.path.join(results_dir, trades_chart_path)):
                html_content.append(f"<h4>Trades Chart</h4>")
                html_content.append(f"<img src='{trades_chart_path}' style='width:100%;max-width:800px;'/>")
            
            equity_path = f"{symbol}_{timeframe}_equity.png"
            if os.path.exists(os.path.join(results_dir, equity_path)):
                html_content.append(f"<h4>Equity Curve</h4>")
                html_content.append(f"<img src='{equity_path}' style='width:100%;max-width:800px;'/>")
            
            drawdown_path = f"{symbol}_{timeframe}_equity_drawdown.png"
            if os.path.exists(os.path.join(results_dir, drawdown_path)):
                html_content.append(f"<h4>Drawdown</h4>")
                html_content.append(f"<img src='{drawdown_path}' style='width:100%;max-width:800px;'/>")
    
    html_content.append("</body></html>")
    
    # Write HTML to file
    if output_file is None:
        output_file = os.path.join(results_dir, "performance_report.html")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(html_content))
    
    logger.info(f"Performance report generated: {output_file}")
    
    return output_file


def find_best_strategy(results):
    """
    Find the best performing strategy based on results.
    
    Parameters:
    -----------
    results : dict
        Dictionary with performance metrics
        
    Returns:
    --------
    tuple
        (symbol, timeframe, performance)
    """
    best_strategy = None
    best_return = float('-inf')
    
    for symbol, timeframe_results in results.items():
        for timeframe, performance in timeframe_results.items():
            # Use a scoring formula: return - 0.5 * abs(drawdown)
            total_return = performance.get('total_return', 0)
            max_drawdown = performance.get('max_drawdown', 0)
            profit_factor = performance.get('profit_factor', 1)
            
            # Simple scoring model
            score = total_return - 0.5 * abs(max_drawdown)
            
            # Add bonus for high profit factor
            if profit_factor > 2:
                score *= 1.2
            
            if score > best_return:
                best_return = score
                best_strategy = (symbol, timeframe, performance)
    
    return best_strategy
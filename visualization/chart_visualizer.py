"""
Visualization module for day trading strategies.
Provides functions to visualize market data, indicators, and trading results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_price_with_indicators(df, save_path=None, title="Price Chart with Indicators"):
    """
    Plot price data with indicators.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data and indicators
    save_path : str
        Path to save the figure
    title : str
        Title of the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot price and MA indicators on the top subplot
    ax1.plot(df.index, df['close'], label='Close Price')
    
    # Plot any available moving averages
    if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
        ax1.plot(df.index, df['sma_fast'], label='Fast SMA', linestyle='--')
        ax1.plot(df.index, df['sma_slow'], label='Slow SMA', linestyle='-.')
    
    # Plot Bollinger Bands if available
    if 'bb_upper' in df.columns and 'bb_middle' in df.columns and 'bb_lower' in df.columns:
        ax1.plot(df.index, df['bb_upper'], 'g--', alpha=0.5, label='Upper BB')
        ax1.plot(df.index, df['bb_middle'], 'g-', alpha=0.5, label='Middle BB')
        ax1.plot(df.index, df['bb_lower'], 'g--', alpha=0.5, label='Lower BB')
    
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Plot RSI if available
    if 'rsi' in df.columns:
        ax2.plot(df.index, df['rsi'], label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)
    
    # Plot MACD if available
    if 'macd_line' in df.columns and 'macd_signal' in df.columns and 'macd_histogram' in df.columns:
        ax3.plot(df.index, df['macd_line'], label='MACD')
        ax3.plot(df.index, df['macd_signal'], label='Signal')
        ax3.bar(df.index, df['macd_histogram'], label='Histogram', alpha=0.5, width=0.6)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True)
    
    ax3.set_xlabel('Date')
    
    # Format the date on the x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_trades(df, save_path=None, title="Trading Strategy Performance"):
    """
    Plot trades with entry and exit points.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with trading signals and results
    save_path : str
        Path to save the figure
    title : str
        Title of the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the price
    ax.plot(df.index, df['close'], label='Close Price')
    
    # Plot entry points
    entries_long = df[(df['position'] == 1) & (df['position'].shift(1) == 0)]
    entries_short = df[(df['position'] == -1) & (df['position'].shift(1) == 0)]
    
    if not entries_long.empty:
        ax.scatter(entries_long.index, entries_long['close'], marker='^', s=100, color='green', label='Long Entry')
    
    if not entries_short.empty:
        ax.scatter(entries_short.index, entries_short['close'], marker='v', s=100, color='red', label='Short Entry')
    
    # Plot exit points
    exits = df[(df['position'] == 0) & ((df['position'].shift(1) == 1) | (df['position'].shift(1) == -1))]
    
    if not exits.empty:
        ax.scatter(exits.index, exits['close'], marker='o', s=100, color='black', label='Exit')
    
    ax.set_title(title)
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    
    # Format the date on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_equity_curve(df, save_path=None, title="Equity Curve"):
    """
    Plot the equity curve.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with trading results including 'capital' column
    save_path : str
        Path to save the figure
    title : str
        Title of the plot
    """
    if 'capital' not in df.columns:
        raise ValueError("DataFrame must contain 'capital' column")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df.index, df['capital'], label='Account Value')
    
    # Calculate and plot drawdown if data is available
    if 'drawdown' not in df.columns and 'capital' in df.columns:
        df['peak'] = df['capital'].cummax()
        df['drawdown'] = (df['capital'] - df['peak']) / df['peak'] * 100
    
    ax.set_title(title)
    ax.set_ylabel('Capital')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    
    # Format the date on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    # Plot drawdown in a separate figure
    if 'drawdown' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
        ax.plot(df.index, df['drawdown'], color='red', label='Drawdown')
        
        ax.set_title("Portfolio Drawdown")
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)
        
        # Format the date on the x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        
        plt.tight_layout()
        
        if save_path:
            drawdown_path = save_path.replace('.png', '_drawdown.png')
            plt.savefig(drawdown_path)
            plt.close()
        else:
            plt.show()


def plot_performance_summary(performance_metrics, save_path=None):
    """
    Plot a summary of performance metrics.
    
    Parameters:
    -----------
    performance_metrics : dict
        Dictionary with performance metrics
    save_path : str
        Path to save the figure
    """
    # Extract metrics
    metrics = {
        'Win Rate': performance_metrics.get('win_rate', 0) * 100,
        'Avg Win (%)': performance_metrics.get('avg_win', 0),
        'Avg Loss (%)': abs(performance_metrics.get('avg_loss', 0)),
        'Profit Factor': performance_metrics.get('profit_factor', 0),
        'Total Return (%)': performance_metrics.get('total_return', 0),
        'Max Drawdown (%)': abs(performance_metrics.get('max_drawdown', 0))
    }
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot win/loss stats
    trades = [
        performance_metrics.get('winning_trades', 0),
        performance_metrics.get('losing_trades', 0)
    ]
    ax1.bar(['Winning Trades', 'Losing Trades'], trades, color=['green', 'red'])
    ax1.set_title('Trade Statistics')
    ax1.set_ylabel('Number of Trades')
    ax1.grid(axis='y')
    
    # Add total trades text
    total_trades = performance_metrics.get('total_trades', 0)
    ax1.text(0.5, 0.9, f"Total Trades: {total_trades}", transform=ax1.transAxes,
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot performance metrics
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    bars = ax2.bar(labels, values)
    
    # Color the bars based on positive/negative values
    for i, bar in enumerate(bars):
        if values[i] < 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    ax2.set_title('Performance Metrics')
    ax2.set_ylabel('Value')
    ax2.grid(axis='y')
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
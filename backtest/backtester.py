"""
Backtesting module for day trading strategies.
Tests the strategy on historical data and calculates performance metrics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.day_trading_strategy import DayTradingStrategy


class DayTradingBacktester:
    """
    Backtester for day trading strategies.
    Tests the strategy on historical data and evaluates its performance.
    """
    
    def __init__(self, strategy=None, initial_capital=10000.0):
        """
        Initialize the backtester with a strategy and initial capital.
        
        Parameters:
        -----------
        strategy : DayTradingStrategy
            The strategy to test
        initial_capital : float
            Initial capital for backtesting
        """
        self.strategy = strategy or DayTradingStrategy()
        self.initial_capital = initial_capital
        self.results = None
        
    def run(self, data, timeframe='15min'):
        """
        Run the backtest on historical data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV data with columns: open, high, low, close, volume
        timeframe : str
            Timeframe of the data (e.g., '15min', '1h')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with backtest results
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Generate signals
        signals_df = self.strategy.generate_signals(df)
        
        # Initialize portfolio columns
        signals_df['position'] = 0  # 0: no position, 1: long, -1: short
        signals_df['entry_price'] = np.nan
        signals_df['stop_loss'] = np.nan
        signals_df['take_profit'] = np.nan
        signals_df['exit_price'] = np.nan
        signals_df['pnl'] = 0.0
        signals_df['capital'] = self.initial_capital
        
        # Simulate trading
        self._simulate_trading(signals_df, timeframe)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(signals_df)
        
        self.results = signals_df
        return signals_df
    
    def _simulate_trading(self, df, timeframe):
        """
        Simulate trading based on signals.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with signals
        timeframe : str
            Timeframe of the data
        """
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        # Calculate time for day trading session
        if timeframe == '15min':
            max_bars_per_day = 32  # Assuming 8-hour trading day
        elif timeframe == '1h':
            max_bars_per_day = 8  # Assuming 8-hour trading day
        else:
            max_bars_per_day = 24  # Default
        
        current_day = None
        day_bars_count = 0
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check if it's a new trading day
            if current_day is None or current_row.name.date() != current_day:
                current_day = current_row.name.date()
                day_bars_count = 0
                # Close position at end of day if still open
                if position != 0:
                    df.loc[prev_row.name, 'position'] = 0
                    df.loc[prev_row.name, 'exit_price'] = prev_row['close']
                    
                    # Calculate PnL
                    if position == 1:  # Long position
                        pnl = (prev_row['close'] - entry_price) / entry_price
                    else:  # Short position
                        pnl = (entry_price - prev_row['close']) / entry_price
                    
                    df.loc[prev_row.name, 'pnl'] = pnl * 100  # Convert to percentage
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
            
            day_bars_count += 1
            
            # Update position based on previous state
            df.loc[current_row.name, 'position'] = position
            
            if position == 0:  # No position, check for new entry
                signal = current_row['signal']
                
                if signal != 0 and day_bars_count < max_bars_per_day - 1:  # Don't enter near end of day
                    position = signal
                    entry_price, stop_loss, take_profit = self.strategy.calculate_entry_exit_prices(df, current_row)
                    
                    df.loc[current_row.name, 'position'] = position
                    df.loc[current_row.name, 'entry_price'] = entry_price
                    df.loc[current_row.name, 'stop_loss'] = stop_loss
                    df.loc[current_row.name, 'take_profit'] = take_profit
            
            else:  # Already in a position, check for exit
                # Check if stop loss or take profit hit
                if position == 1:  # Long position
                    if current_row['low'] <= stop_loss:  # Stop loss hit
                        df.loc[current_row.name, 'position'] = 0
                        df.loc[current_row.name, 'exit_price'] = stop_loss
                        
                        # Calculate PnL
                        pnl = (stop_loss - entry_price) / entry_price
                        df.loc[current_row.name, 'pnl'] = pnl * 100  # Convert to percentage
                        
                        position = 0
                    
                    elif current_row['high'] >= take_profit:  # Take profit hit
                        df.loc[current_row.name, 'position'] = 0
                        df.loc[current_row.name, 'exit_price'] = take_profit
                        
                        # Calculate PnL
                        pnl = (take_profit - entry_price) / entry_price
                        df.loc[current_row.name, 'pnl'] = pnl * 100  # Convert to percentage
                        
                        position = 0
                
                elif position == -1:  # Short position
                    if current_row['high'] >= stop_loss:  # Stop loss hit
                        df.loc[current_row.name, 'position'] = 0
                        df.loc[current_row.name, 'exit_price'] = stop_loss
                        
                        # Calculate PnL
                        pnl = (entry_price - stop_loss) / entry_price
                        df.loc[current_row.name, 'pnl'] = pnl * 100  # Convert to percentage
                        
                        position = 0
                    
                    elif current_row['low'] <= take_profit:  # Take profit hit
                        df.loc[current_row.name, 'position'] = 0
                        df.loc[current_row.name, 'exit_price'] = take_profit
                        
                        # Calculate PnL
                        pnl = (entry_price - take_profit) / entry_price
                        df.loc[current_row.name, 'pnl'] = pnl * 100  # Convert to percentage
                        
                        position = 0
                
                # Check if opposite signal received and exit
                if (position == 1 and current_row['signal'] == -1) or (position == -1 and current_row['signal'] == 1):
                    df.loc[current_row.name, 'position'] = 0
                    df.loc[current_row.name, 'exit_price'] = current_row['close']
                    
                    # Calculate PnL
                    if position == 1:  # Long position
                        pnl = (current_row['close'] - entry_price) / entry_price
                    else:  # Short position
                        pnl = (entry_price - current_row['close']) / entry_price
                    
                    df.loc[current_row.name, 'pnl'] = pnl * 100  # Convert to percentage
                    
                    position = 0
        
        # Calculate cumulative capital
        cumulative_pnl = 0
        for i in range(len(df)):
            if df.iloc[i]['pnl'] != 0:
                trade_pnl = df.iloc[i]['pnl'] / 100  # Convert percentage to decimal
                capital_change = self.initial_capital * trade_pnl
                cumulative_pnl += capital_change
            
            df.iloc[i, df.columns.get_loc('capital')] = self.initial_capital + cumulative_pnl
    
    def _calculate_performance_metrics(self, df):
        """
        Calculate performance metrics for the strategy.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with trading results
        """
        # Filter completed trades (non-zero PnL)
        trades = df[df['pnl'] != 0].copy()
        
        # Calculate metrics
        self.total_trades = len(trades)
        self.winning_trades = len(trades[trades['pnl'] > 0])
        self.losing_trades = len(trades[trades['pnl'] < 0])
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
            self.avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if self.winning_trades > 0 else 0
            self.avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if self.losing_trades > 0 else 0
            self.profit_factor = abs(trades[trades['pnl'] > 0]['pnl'].sum() / trades[trades['pnl'] < 0]['pnl'].sum()) if trades[trades['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
            
            # Calculate final capital and return
            initial_capital = df.iloc[0]['capital']
            final_capital = df.iloc[-1]['capital']
            self.total_return = ((final_capital - initial_capital) / initial_capital) * 100
            
            # Calculate drawdown
            df['peak'] = df['capital'].cummax()
            df['drawdown'] = (df['capital'] - df['peak']) / df['peak'] * 100
            self.max_drawdown = df['drawdown'].min()
        else:
            self.win_rate = 0
            self.avg_win = 0
            self.avg_loss = 0
            self.profit_factor = 0
            self.total_return = 0
            self.max_drawdown = 0
    
    def get_performance_summary(self):
        """
        Get a summary of the backtest performance.
        
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        if self.results is None:
            return {'error': 'Backtest not run yet'}
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown
        }
    
    def plot_results(self, save_path=None):
        """
        Plot the results of the backtest.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot to
        """
        if self.results is None:
            print("Backtest not run yet")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # Plot price and indicators
        ax1.plot(self.results.index, self.results['close'], label='Price')
        ax1.plot(self.results.index, self.results['sma_fast'], label=f"SMA ({self.strategy.config['sma_fast_period']})")
        ax1.plot(self.results.index, self.results['sma_slow'], label=f"SMA ({self.strategy.config['sma_slow_period']})")
        ax1.plot(self.results.index, self.results['bb_upper'], 'g--', label='BB Upper')
        ax1.plot(self.results.index, self.results['bb_middle'], 'g-', label='BB Middle')
        ax1.plot(self.results.index, self.results['bb_lower'], 'g--', label='BB Lower')
        
        # Plot buy and sell points
        buy_signals = self.results[self.results['signal'] == 1]
        sell_signals = self.results[self.results['signal'] == -1]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal')
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal')
        
        # Plot exit points
        exits = self.results[self.results['exit_price'].notna()]
        
        if not exits.empty:
            ax1.scatter(exits.index, exits['exit_price'], marker='o', color='black', label='Exit')
        
        ax1.set_title('Price, Indicators, and Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot RSI
        ax2.plot(self.results.index, self.results['rsi'], label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)
        
        # Plot MACD
        ax3.plot(self.results.index, self.results['macd_line'], label='MACD Line')
        ax3.plot(self.results.index, self.results['macd_signal'], label='Signal Line')
        ax3.bar(self.results.index, self.results['macd_histogram'], label='Histogram', alpha=0.5)
        ax3.set_title('MACD')
        ax3.set_ylabel('MACD')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['capital'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.grid(True)
        
        if save_path:
            equity_save_path = save_path.replace('.png', '_equity.png')
            plt.savefig(equity_save_path)
            plt.close()
        else:
            plt.show()
            
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['drawdown'])
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        if save_path:
            drawdown_save_path = save_path.replace('.png', '_drawdown.png')
            plt.savefig(drawdown_save_path)
            plt.close()
        else:
            plt.show()
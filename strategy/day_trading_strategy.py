"""
Core day trading strategy implementation.
Uses technical indicators to generate buy/sell signals.
"""
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.technical_indicators import (
    sma, ema, bollinger_bands, rsi, macd
)


class DayTradingStrategy:
    """
    Day trading strategy that generates signals based on technical indicators.
    Strategy focuses on intraday trading on M15 and H1 timeframes.
    """
    
    def __init__(self, config=None):
        """
        Initialize the day trading strategy with configuration parameters.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary with strategy parameters
        """
        # Default configuration
        self.config = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'sma_fast_period': 20,
            'sma_slow_period': 50,
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9,
            'stop_loss_pct': 0.5,  # 0.5% stop loss
            'take_profit_pct': 1.0,  # 1.0% take profit
        }
        
        # Update config with user-provided values
        if config:
            self.config.update(config)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy rules.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV data with columns: open, high, low, close, volume
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added signal columns
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Calculate indicators
        df['sma_fast'] = sma(df['close'], self.config['sma_fast_period'])
        df['sma_slow'] = sma(df['close'], self.config['sma_slow_period'])
        
        upper_band, middle_band, lower_band = bollinger_bands(
            df['close'], 
            self.config['bb_period'],
            self.config['bb_std']
        )
        df['bb_upper'] = upper_band
        df['bb_middle'] = middle_band
        df['bb_lower'] = lower_band
        
        df['rsi'] = rsi(df['close'], self.config['rsi_period'])
        
        macd_line, signal_line, histogram = macd(
            df['close'],
            self.config['macd_fast_period'],
            self.config['macd_slow_period'],
            self.config['macd_signal_period']
        )
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Initialize signal columns
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        df['signal_reason'] = ''
        
        # Generate signals based on strategy rules
        self._generate_buy_signals(df)
        self._generate_sell_signals(df)
        
        return df
    
    def _generate_buy_signals(self, df):
        """
        Generate buy signals based on the strategy rules.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with calculated indicators
        """
        # Rule 1: RSI oversold + price near lower Bollinger Band
        condition1 = (
            (df['rsi'] < self.config['rsi_oversold']) & 
            (df['close'] <= df['bb_lower'] * 1.01)  # Close to or below lower BB
        )
        
        # Rule 2: MACD line crosses above signal line + positive momentum
        condition2 = (
            (df['macd_line'] > df['macd_signal']) & 
            (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)) &
            (df['macd_histogram'] > 0)
        )
        
        # Rule 3: Fast SMA crosses above slow SMA
        condition3 = (
            (df['sma_fast'] > df['sma_slow']) & 
            (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1))
        )
        
        # Combined buy signals
        df.loc[condition1, 'signal'] = 1
        df.loc[condition1, 'signal_reason'] = 'RSI_OVERSOLD_BB_LOWER'
        
        df.loc[condition2 & (df['signal'] == 0), 'signal'] = 1
        df.loc[condition2 & (df['signal'] == 1), 'signal_reason'] = 'MACD_CROSSOVER'
        
        df.loc[condition3 & (df['signal'] == 0), 'signal'] = 1
        df.loc[condition3 & (df['signal'] == 1), 'signal_reason'] = 'SMA_CROSSOVER'
    
    def _generate_sell_signals(self, df):
        """
        Generate sell signals based on the strategy rules.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with calculated indicators
        """
        # Rule 1: RSI overbought + price near upper Bollinger Band
        condition1 = (
            (df['rsi'] > self.config['rsi_overbought']) & 
            (df['close'] >= df['bb_upper'] * 0.99)  # Close to or above upper BB
        )
        
        # Rule 2: MACD line crosses below signal line + negative momentum
        condition2 = (
            (df['macd_line'] < df['macd_signal']) & 
            (df['macd_line'].shift(1) >= df['macd_signal'].shift(1)) &
            (df['macd_histogram'] < 0)
        )
        
        # Rule 3: Fast SMA crosses below slow SMA
        condition3 = (
            (df['sma_fast'] < df['sma_slow']) & 
            (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1))
        )
        
        # Combined sell signals
        df.loc[condition1, 'signal'] = -1
        df.loc[condition1, 'signal_reason'] = 'RSI_OVERBOUGHT_BB_UPPER'
        
        df.loc[condition2 & (df['signal'] == 0), 'signal'] = -1
        df.loc[condition2 & (df['signal'] == -1), 'signal_reason'] = 'MACD_CROSSOVER'
        
        df.loc[condition3 & (df['signal'] == 0), 'signal'] = -1
        df.loc[condition3 & (df['signal'] == -1), 'signal_reason'] = 'SMA_CROSSOVER'
    
    def calculate_entry_exit_prices(self, data, signal_row):
        """
        Calculate entry, stop loss, and take profit prices for a signal.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV data
        signal_row : pandas.Series
            Row from the DataFrame containing the signal
            
        Returns:
        --------
        tuple
            (entry_price, stop_loss, take_profit)
        """
        signal = signal_row['signal']
        entry_price = signal_row['close']
        
        if signal == 1:  # Buy signal
            stop_loss = entry_price * (1 - self.config['stop_loss_pct'] / 100)
            take_profit = entry_price * (1 + self.config['take_profit_pct'] / 100)
        elif signal == -1:  # Sell signal
            stop_loss = entry_price * (1 + self.config['stop_loss_pct'] / 100)
            take_profit = entry_price * (1 - self.config['take_profit_pct'] / 100)
        else:
            return None, None, None
        
        return entry_price, stop_loss, take_profit
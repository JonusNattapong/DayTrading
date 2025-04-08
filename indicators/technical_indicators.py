"""
Technical indicators module for day trading strategy.
Implements Moving Averages, Bollinger Bands, RSI, and MACD indicators.
"""
import numpy as np
import pandas as pd


def sma(data, period=14):
    """
    Calculate Simple Moving Average
    
    Parameters:
    -----------
    data : pandas.Series
        Price data, typically close prices
    period : int
        The period over which to calculate the indicator
        
    Returns:
    --------
    pandas.Series
        Simple Moving Average values
    """
    return data.rolling(window=period).mean()


def ema(data, period=14):
    """
    Calculate Exponential Moving Average
    
    Parameters:
    -----------
    data : pandas.Series
        Price data, typically close prices
    period : int
        The period over which to calculate the indicator
        
    Returns:
    --------
    pandas.Series
        Exponential Moving Average values
    """
    return data.ewm(span=period, adjust=False).mean()


def bollinger_bands(data, period=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Parameters:
    -----------
    data : pandas.Series
        Price data, typically close prices
    period : int
        The period over which to calculate the moving average
    num_std : int
        Number of standard deviations for the bands
        
    Returns:
    --------
    tuple of pandas.Series
        (upper_band, middle_band, lower_band)
    """
    middle_band = sma(data, period)
    std_dev = data.rolling(window=period).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return upper_band, middle_band, lower_band


def rsi(data, period=14):
    """
    Calculate Relative Strength Index
    
    Parameters:
    -----------
    data : pandas.Series
        Price data, typically close prices
    period : int
        The period over which to calculate the indicator
        
    Returns:
    --------
    pandas.Series
        RSI values
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence
    
    Parameters:
    -----------
    data : pandas.Series
        Price data, typically close prices
    fast_period : int
        The period for the fast EMA
    slow_period : int
        The period for the slow EMA
    signal_period : int
        The period for the signal line
        
    Returns:
    --------
    tuple of pandas.Series
        (macd_line, signal_line, histogram)
    """
    # Calculate fast and slow EMAs
    fast_ema = ema(data, fast_period)
    slow_ema = ema(data, slow_period)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = ema(macd_line, signal_period)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram
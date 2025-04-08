"""
Data handling module for day trading strategies.
Provides functions to download, process, and manage financial market data.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_data(symbol, start_date, end_date, interval='15m', save=True, data_dir=None):
    """
    Download historical market data for a given symbol.
    
    Parameters:
    -----------
    symbol : str
        The ticker symbol to download data for
    start_date : str or datetime
        The start date for the data
    end_date : str or datetime
        The end date for the data
    interval : str
        The timeframe of the data ('15m', '1h', etc.)
    save : bool
        Whether to save the data to a file
    data_dir : str
        Directory to save the data to
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with OHLCV data
    """
    logger.info(f"Downloading {interval} data for {symbol} from {start_date} to {end_date}")
    
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        
        # Handle empty data
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Save data if requested
        if save:
            if data_dir is None:
                # Default to 'data' directory in the current project
                data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            
            # Create directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Create a filename with the symbol, interval, and date range
            if isinstance(start_date, str):
                start_str = start_date.replace('-', '')
            else:
                start_str = start_date.strftime('%Y%m%d')
            
            if isinstance(end_date, str):
                end_str = end_date.replace('-', '')
            else:
                end_str = end_date.strftime('%Y%m%d')
                
            filename = f"{symbol}_{interval}_{start_str}_{end_str}.csv"
            file_path = os.path.join(data_dir, filename)
            
            # Save to CSV
            df.to_csv(file_path)
            logger.info(f"Data saved to {file_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {str(e)}")
        return None


def load_data(file_path):
    """
    Load market data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with OHLCV data
    """
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return None


def process_data_for_day_trading(df, trading_hours=None):
    """
    Process data for day trading, filtering for trading hours.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV data
    trading_hours : tuple
        Tuple of (start_hour, end_hour) for trading hours (24h format)
        
    Returns:
    --------
    pandas.DataFrame
        Processed data
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Filter for trading hours if specified
    if trading_hours:
        start_hour, end_hour = trading_hours
        df_processed = df_processed.between_time(f'{start_hour:02d}:00', f'{end_hour:02d}:00')
    
    # Ensure we have all required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df_processed.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        return None
    
    return df_processed


def get_daily_bars(df, interval='15m'):
    """
    Group data into daily bars.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV data with higher time resolution
    interval : str
        The timeframe of the input data
        
    Returns:
    --------
    pandas.DataFrame
        Daily OHLCV data
    """
    daily_bars = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    return daily_bars.dropna()


def get_multiple_symbols_data(symbols, start_date, end_date, interval='15m', save=True, data_dir=None):
    """
    Download data for multiple symbols.
    
    Parameters:
    -----------
    symbols : list
        List of ticker symbols
    start_date : str or datetime
        The start date for the data
    end_date : str or datetime
        The end date for the data
    interval : str
        The timeframe of the data
    save : bool
        Whether to save the data to files
    data_dir : str
        Directory to save the data to
        
    Returns:
    --------
    dict
        Dictionary with DataFrames for each symbol
    """
    data_dict = {}
    
    for symbol in symbols:
        df = download_data(symbol, start_date, end_date, interval, save, data_dir)
        if df is not None:
            data_dict[symbol] = df
    
    return data_dict
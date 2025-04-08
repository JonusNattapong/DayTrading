# Day Trading Strategy

## Overview
This repository contains a day trading strategy implementation that focuses on intraday trading without holding positions overnight.

## Strategy Details
- **Type**: Day Trading (Intraday)
- **Timeframes**: M15, H1
- **Position Duration**: Positions opened and closed within the same trading day
- **Risk Management**: Strict stop loss and take profit levels

## Technical Indicators Used
- Moving Averages (Simple and Exponential)
- Bollinger Bands
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)

## Implementation
The strategy is implemented in Python using libraries such as Pandas, NumPy, and Matplotlib for data analysis and visualization.

## Project Structure
- `strategy/`: Core strategy implementation
- `indicators/`: Technical indicators implementation
- `backtest/`: Backtesting framework
- `data/`: Market data storage
- `visualization/`: Trading charts and performance visualization
- `config/`: Configuration files

## Getting Started
1. Install the required dependencies
2. Configure your trading parameters
3. Run the backtesting to validate the strategy
4. Deploy for paper trading or live trading

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- ta-lib (Technical Analysis Library)
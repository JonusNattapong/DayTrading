# Day Trading Strategy Backtester

A comprehensive day trading strategy backtester and analysis toolkit for developing and testing technical analysis-based trading strategies.

## Overview

This project provides a framework for backtesting day trading strategies using technical indicators like Moving Averages, Bollinger Bands, RSI, and MACD. It allows traders to develop, test, and optimize their trading strategies with historical data before risking real capital.

## Features

- **Data Management**: Download and process historical financial data
- **Technical Indicators**: Implementation of popular indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- **Strategy Development**: Framework for creating custom day trading strategies
- **Backtesting Engine**: Rigorous backtesting with realistic trading conditions
- **Performance Analysis**: Comprehensive performance metrics and trade analytics
- **Strategy Optimization**: Parameter optimization using grid search
- **Visualization**: Charts and plots for prices, indicators, trades, and performance
- **Reporting**: Detailed HTML reports with performance metrics and visualizations
- **Jupyter Integration**: Interactive analysis with Jupyter notebooks

## Project Structure

```
DayTrading/
├── backtest/           # Backtesting engine
├── config/             # Configuration files
├── data/               # Data handling modules and storage
├── indicators/         # Technical indicators implementation
├── notebooks/          # Jupyter notebooks for analysis
├── strategy/           # Trading strategy implementation
├── utils/              # Utility functions and analysis tools
├── visualization/      # Visualization tools
├── main.py             # Main application entry point
├── optimize.py         # Strategy optimization CLI
├── generate_report.py  # Report generation CLI
├── setup.py            # Project setup script
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DayTrading.git
cd DayTrading
```

2. Run the setup script:
```bash
python setup.py
```

Or install dependencies manually:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Edit the configuration file in `config/config.json` to set your backtesting parameters:

```json
{
  "strategy": {
    "timeframes": ["15m", "1h"],
    "indicators": {
      "sma_fast_period": 20,
      "sma_slow_period": 50,
      "rsi_period": 14,
      "rsi_oversold": 30,
      "rsi_overbought": 70,
      "bb_period": 20,
      "bb_std": 2.0,
      "macd_fast_period": 12,
      "macd_slow_period": 26,
      "macd_signal_period": 9
    },
    "risk_management": {
      "stop_loss_pct": 1.0,
      "take_profit_pct": 2.0,
      "max_risk_per_trade_pct": 1.0
    }
  },
  "backtest": {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 10000.0,
    "trading_hours": {
      "start": 9,
      "end": 16
    }
  },
  "data": {
    "data_dir": "data",
    "use_cached_data": true
  },
  "output": {
    "results_dir": "results",
    "save_results": true,
    "save_trades": true,
    "plot_charts": true
  }
}
```

### Running a Backtest

Run a backtest with the default configuration:
```bash
python main.py
```

Or specify a custom configuration file:
```bash
python main.py --config path/to/your/config.json
```

### Optimizing Strategy Parameters

Optimize strategy parameters for a specific symbol and timeframe:
```bash
python optimize.py --symbol AAPL --timeframe 15m --max-combinations 100 --scoring combined
```

### Generating Reports

Generate a comprehensive HTML report from backtest results:
```bash
python generate_report.py --results-dir results/run_20230101_120000
```

### Interactive Analysis

Use the provided Jupyter notebook for interactive analysis:
```bash
jupyter notebook notebooks/day_trading_analysis.ipynb
```

## Customizing Strategies

To create your own trading strategy, you can extend the `DayTradingStrategy` class in `strategy/day_trading_strategy.py`:

```python
class MyCustomStrategy(DayTradingStrategy):
    def __init__(self, params):
        super().__init__(params)
        # Initialize your custom parameters
        
    def generate_signals(self, data):
        # Implement your custom signal generation logic
        return signals
```

## Performance Metrics

The backtester calculates the following performance metrics:
- Total Return (%)
- Win Rate (%)
- Average Win (%)
- Average Loss (%)
- Profit Factor
- Maximum Drawdown (%)
- Sharpe Ratio
- Total Trades
- Winning Trades
- Losing Trades

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- Jupyter (optional, for notebooks)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
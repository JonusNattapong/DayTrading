"""
Strategy optimizer module for day trading strategies.
Provides functionality to optimize strategy parameters.
"""
import os
import sys
import json
import logging
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.day_trading_strategy import DayTradingStrategy
from backtest.backtester import DayTradingBacktester

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Optimizer for day trading strategy parameters.
    Uses grid search to find optimal parameters.
    """
    
    def __init__(self, data, timeframe='15m', initial_capital=10000.0):
        """
        Initialize the optimizer with data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV data
        timeframe : str
            Timeframe of the data
        initial_capital : float
            Initial capital for backtesting
        """
        self.data = data
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.results = []
    
    def optimize(self, param_grid, scoring='total_return', max_combinations=100):
        """
        Optimize strategy parameters using grid search.
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary with parameter names and possible values
        scoring : str or callable
            Metric to optimize (default: 'total_return')
            If callable, should take performance dict and return a score (higher is better)
        max_combinations : int
            Maximum number of parameter combinations to test
            
        Returns:
        --------
        tuple
            (best_params, best_score, all_results)
        """
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        
        # Limit the number of combinations if needed
        if len(combinations) > max_combinations:
            logger.warning(f"Limiting to {max_combinations} parameter combinations")
            np.random.seed(42)
            indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]
        
        # Initialize results
        self.results = []
        best_score = float('-inf')
        best_params = None
        
        # Run backtests for each parameter combination
        for i, values in enumerate(tqdm(combinations, desc="Optimizing parameters")):
            # Create parameter dictionary
            params = dict(zip(param_names, values))
            
            # Run backtest with these parameters
            try:
                performance = self._backtest_with_params(params)
                
                # Calculate score
                if callable(scoring):
                    score = scoring(performance)
                else:
                    # Default to using the specified metric
                    score = performance.get(scoring, float('-inf'))
                
                # Track results
                result = {
                    'params': params,
                    'performance': performance,
                    'score': score
                }
                self.results.append(result)
                
                # Check if this is the best so far
                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"New best score: {best_score} with params: {best_params}")
            
            except Exception as e:
                logger.error(f"Error in backtest with params {params}: {str(e)}")
        
        # Sort results by score
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        return best_params, best_score, self.results
    
    def _backtest_with_params(self, params):
        """
        Run a backtest with specified parameters.
        
        Parameters:
        -----------
        params : dict
            Strategy parameters
            
        Returns:
        --------
        dict
            Performance metrics
        """
        # Create strategy with parameters
        strategy = DayTradingStrategy(params)
        
        # Create backtester
        backtester = DayTradingBacktester(strategy, self.initial_capital)
        
        # Run backtest
        backtester.run(self.data, self.timeframe)
        
        # Get performance
        performance = backtester.get_performance_summary()
        
        return performance
    
    def get_top_n_strategies(self, n=10):
        """
        Get the top N strategies by score.
        
        Parameters:
        -----------
        n : int
            Number of top strategies to return
            
        Returns:
        --------
        list
            List of top N strategies
        """
        return self.results[:n]
    
    def save_results(self, file_path):
        """
        Save optimization results to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save results to
        """
        # Create a serializable version of the results
        serializable_results = []
        
        for result in self.results:
            serializable_result = {
                'params': result['params'],
                'performance': result['performance'],
                'score': result['score']
            }
            serializable_results.append(serializable_result)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Optimization results saved to {file_path}")
    
    def generate_report(self, file_path, top_n=10):
        """
        Generate a report of the optimization results.
        
        Parameters:
        -----------
        file_path : str
            Path to save the report to
        top_n : int
            Number of top strategies to include in the report
        """
        if not self.results:
            logger.warning("No optimization results to report")
            return
        
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
        html_content.append("</style>")
        html_content.append("<title>Strategy Optimization Report</title>")
        html_content.append("</head><body>")
        
        # Add header
        html_content.append("<h1>Strategy Optimization Report</h1>")
        html_content.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Add top strategies
        html_content.append(f"<h2>Top {top_n} Strategies</h2>")
        
        # Create table
        html_content.append("<table>")
        html_content.append("<tr><th>Rank</th><th>Score</th><th>Parameters</th><th>Performance</th></tr>")
        
        for i, result in enumerate(self.results[:top_n]):
            params_str = "<br>".join([f"{k}: {v}" for k, v in result['params'].items()])
            
            perf = result['performance']
            perf_str = "<br>".join([
                f"Total Trades: {perf.get('total_trades', 0)}",
                f"Win Rate: {perf.get('win_rate', 0) * 100:.2f}%",
                f"Total Return: {perf.get('total_return', 0):.2f}%",
                f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}%",
                f"Profit Factor: {perf.get('profit_factor', 0):.2f}"
            ])
            
            html_content.append(f"<tr><td>{i+1}</td><td>{result['score']:.2f}</td><td>{params_str}</td><td>{perf_str}</td></tr>")
        
        html_content.append("</table>")
        
        # Add parameter analysis
        html_content.append("<h2>Parameter Analysis</h2>")
        
        # Get parameter names from first result
        if self.results:
            param_names = list(self.results[0]['params'].keys())
            
            for param in param_names:
                html_content.append(f"<h3>Effect of {param}</h3>")
                
                # Group results by this parameter value
                param_values = {}
                
                for result in self.results:
                    value = result['params'][param]
                    if value not in param_values:
                        param_values[value] = []
                    param_values[value].append(result['score'])
                
                # Calculate average score for each value
                avg_scores = {value: sum(scores) / len(scores) for value, scores in param_values.items()}
                
                # Sort by value
                sorted_values = sorted(avg_scores.items())
                
                # Create a simple bar chart (ASCII-style)
                max_score = max(avg_scores.values())
                width = 50  # max width of the bar
                
                html_content.append("<pre>")
                for value, score in sorted_values:
                    bar_length = int((score / max_score) * width)
                    bar = "█" * bar_length
                    html_content.append(f"{value:<10} {score:>8.2f} |{bar}")
                html_content.append("</pre>")
        
        html_content.append("</body></html>")
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write("\n".join(html_content))
        
        logger.info(f"Optimization report saved to {file_path}")


def optimize_strategy(data, param_grid, timeframe='15m', initial_capital=10000.0, 
                     scoring='total_return', max_combinations=100):
    """
    Convenience function to optimize a strategy.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        OHLCV data
    param_grid : dict
        Dictionary with parameter names and possible values
    timeframe : str
        Timeframe of the data
    initial_capital : float
        Initial capital for backtesting
    scoring : str or callable
        Metric to optimize
    max_combinations : int
        Maximum number of parameter combinations to test
        
    Returns:
    --------
    tuple
        (best_params, best_score, optimizer)
    """
    optimizer = StrategyOptimizer(data, timeframe, initial_capital)
    best_params, best_score, _ = optimizer.optimize(param_grid, scoring, max_combinations)
    
    return best_params, best_score, optimizer
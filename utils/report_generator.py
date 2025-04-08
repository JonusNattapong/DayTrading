"""
Report generator module for day trading strategy backtest results.
Generates comprehensive HTML reports with performance metrics and visualizations.
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analysis import load_results, load_trade_data, analyze_trades, compare_timeframes

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generator for comprehensive backtest reports.
    Creates HTML reports with performance metrics and visualizations.
    """
    
    def __init__(self, results_dir, output_path=None):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        results_dir : str
            Path to the directory containing backtest results
        output_path : str
            Path to save the HTML report to
        """
        self.results_dir = results_dir
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"backtest_report_{timestamp}.html")
        
        self.output_path = output_path
        self.results = load_results(results_dir)
        
        # For storing plots
        self.plot_images = {}
    
    def generate_report(self):
        """
        Generate a comprehensive HTML report.
        
        Returns:
        --------
        str
            Path to the generated report
        """
        if not self.results:
            logger.error("No results to generate report from")
            return None
        
        # Create HTML content
        html_content = []
        html_content.append(self._create_html_header())
        
        # Add summary section
        html_content.append(self._create_summary_section())
        
        # Add comparison section
        html_content.append(self._create_comparison_section())
        
        # Add detailed sections for each symbol and timeframe
        html_content.append(self._create_detailed_sections())
        
        # Add trades analysis section
        html_content.append(self._create_trades_analysis_section())
        
        # Add footer
        html_content.append(self._create_html_footer())
        
        # Write to file
        with open(self.output_path, 'w') as f:
            f.write("\n".join(html_content))
        
        logger.info(f"Report generated: {self.output_path}")
        return self.output_path
    
    def _create_html_header(self):
        """Create HTML header."""
        header = []
        header.append("<!DOCTYPE html>")
        header.append("<html lang='en'>")
        header.append("<head>")
        header.append("    <meta charset='UTF-8'>")
        header.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        header.append("    <title>Day Trading Strategy Backtest Report</title>")
        header.append("    <style>")
        header.append("        body { font-family: Arial, sans-serif; margin: 20px; color: #333; }")
        header.append("        h1, h2, h3, h4 { color: #2c3e50; }")
        header.append("        .container { max-width: 1200px; margin: 0 auto; }")
        header.append("        .summary-box { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; background-color: #f9f9f9; }")
        header.append("        .metric-box { display: inline-block; width: 180px; text-align: center; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: white; }")
        header.append("        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }")
        header.append("        .metric-label { font-size: 14px; color: #666; }")
        header.append("        .good { color: green; }")
        header.append("        .bad { color: red; }")
        header.append("        .neutral { color: #2c3e50; }")
        header.append("        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        header.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        header.append("        th { background-color: #f2f2f2; }")
        header.append("        tr:nth-child(even) { background-color: #f9f9f9; }")
        header.append("        .chart-container { margin: 20px 0; }")
        header.append("        .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }")
        header.append("        .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }")
        header.append("        .tab button:hover { background-color: #ddd; }")
        header.append("        .tab button.active { background-color: #ccc; }")
        header.append("        .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }")
        header.append("    </style>")
        header.append("    <script>")
        header.append("        function openTab(evt, tabName) {")
        header.append("            var i, tabcontent, tablinks;")
        header.append("            tabcontent = document.getElementsByClassName('tabcontent');")
        header.append("            for (i = 0; i < tabcontent.length; i++) {")
        header.append("                tabcontent[i].style.display = 'none';")
        header.append("            }")
        header.append("            tablinks = document.getElementsByClassName('tablinks');")
        header.append("            for (i = 0; i < tablinks.length; i++) {")
        header.append("                tablinks[i].className = tablinks[i].className.replace(' active', '');")
        header.append("            }")
        header.append("            document.getElementById(tabName).style.display = 'block';")
        header.append("            evt.currentTarget.className += ' active';")
        header.append("        }")
        header.append("    </script>")
        header.append("</head>")
        header.append("<body>")
        header.append("<div class='container'>")
        header.append(f"<h1>Day Trading Strategy Backtest Report</h1>")
        header.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        return "\n".join(header)
    
    def _create_html_footer(self):
        """Create HTML footer."""
        footer = []
        footer.append("</div>") # Close container
        footer.append("</body>")
        footer.append("</html>")
        
        return "\n".join(footer)
    
    def _create_summary_section(self):
        """Create summary section with overall performance metrics."""
        section = []
        section.append("<h2>Summary</h2>")
        section.append("<div class='summary-box'>")
        
        # Calculate overall metrics
        total_trades = 0
        winning_trades = 0
        total_return = 0
        symbols_count = 0
        timeframes_count = 0
        
        for symbol, timeframe_results in self.results.items():
            symbols_count += 1
            for timeframe, performance in timeframe_results.items():
                timeframes_count += 1
                total_trades += performance.get('total_trades', 0)
                winning_trades += performance.get('winning_trades', 0)
                total_return += performance.get('total_return', 0)
        
        # Calculate averages
        avg_return = total_return / timeframes_count if timeframes_count > 0 else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Add metric boxes
        section.append("<div class='metric-box'>")
        section.append(f"<div class='metric-label'>Total Trades</div>")
        section.append(f"<div class='metric-value neutral'>{total_trades}</div>")
        section.append("</div>")
        
        section.append("<div class='metric-box'>")
        section.append(f"<div class='metric-label'>Win Rate</div>")
        color = "good" if win_rate >= 0.5 else "bad"
        section.append(f"<div class='metric-value {color}'>{win_rate:.2%}</div>")
        section.append("</div>")
        
        section.append("<div class='metric-box'>")
        section.append(f"<div class='metric-label'>Avg Return</div>")
        color = "good" if avg_return >= 0 else "bad"
        section.append(f"<div class='metric-value {color}'>{avg_return:.2f}%</div>")
        section.append("</div>")
        
        section.append("<div class='metric-box'>")
        section.append(f"<div class='metric-label'>Symbols Tested</div>")
        section.append(f"<div class='metric-value neutral'>{symbols_count}</div>")
        section.append("</div>")
        
        section.append("<div class='metric-box'>")
        section.append(f"<div class='metric-label'>Timeframes Tested</div>")
        section.append(f"<div class='metric-value neutral'>{timeframes_count}</div>")
        section.append("</div>")
        
        section.append("</div>") # Close summary-box
        
        return "\n".join(section)
    
    def _create_comparison_section(self):
        """Create section with comparison across symbols and timeframes."""
        section = []
        section.append("<h2>Strategy Comparison</h2>")
        
        # Create comparison table using analysis function
        comparison_df = compare_timeframes(self.results)
        
        if not comparison_df.empty:
            # Convert to HTML table
            section.append(comparison_df.to_html(index=False))
            
            # Create a plot for comparison
            img_base64 = self._create_comparison_plot(comparison_df)
            if img_base64:
                section.append("<div class='chart-container'>")
                section.append(f"<img src='data:image/png;base64,{img_base64}' style='width:100%;max-width:1000px;'>")
                section.append("</div>")
        else:
            section.append("<p>No data available for comparison.</p>")
        
        return "\n".join(section)
    
    def _create_detailed_sections(self):
        """Create detailed sections for each symbol and timeframe."""
        section = []
        section.append("<h2>Detailed Results</h2>")
        
        # Create tabs for different symbols
        section.append("<div class='tab'>")
        for i, symbol in enumerate(self.results.keys()):
            active = "active" if i == 0 else ""
            section.append(f"<button class='tablinks {active}' onclick=\"openTab(event, '{symbol}')\">{symbol}</button>")
        section.append("</div>")
        
        # Create tab content for each symbol
        for i, (symbol, timeframe_results) in enumerate(self.results.items()):
            display = "block" if i == 0 else "none"
            section.append(f"<div id='{symbol}' class='tabcontent' style='display:{display};'>")
            section.append(f"<h3>{symbol}</h3>")
            
            # Create inner tabs for timeframes
            section.append("<div class='tab'>")
            for j, timeframe in enumerate(timeframe_results.keys()):
                active = "active" if j == 0 else ""
                tab_id = f"{symbol}_{timeframe}"
                section.append(f"<button class='tablinks {active}' onclick=\"openTab(event, '{tab_id}')\">{timeframe}</button>")
            section.append("</div>")
            
            # Create tab content for each timeframe
            for j, (timeframe, performance) in enumerate(timeframe_results.items()):
                display = "block" if j == 0 else "none"
                tab_id = f"{symbol}_{timeframe}"
                section.append(f"<div id='{tab_id}' class='tabcontent' style='display:{display};'>")
                section.append(f"<h4>Timeframe: {timeframe}</h4>")
                
                # Add performance metrics
                section.append("<div class='summary-box'>")
                metrics = [
                    ("Total Trades", performance.get('total_trades', 0), "neutral"),
                    ("Win Rate", f"{performance.get('win_rate', 0) * 100:.2f}%", "good" if performance.get('win_rate', 0) >= 0.5 else "bad"),
                    ("Total Return", f"{performance.get('total_return', 0):.2f}%", "good" if performance.get('total_return', 0) >= 0 else "bad"),
                    ("Max Drawdown", f"{performance.get('max_drawdown', 0):.2f}%", "bad" if performance.get('max_drawdown', 0) < 0 else "neutral"),
                    ("Profit Factor", f"{performance.get('profit_factor', 0):.2f}", "good" if performance.get('profit_factor', 0) > 1 else "bad")
                ]
                
                # Create metric boxes
                for label, value, color in metrics:
                    section.append("<div class='metric-box'>")
                    section.append(f"<div class='metric-label'>{label}</div>")
                    section.append(f"<div class='metric-value {color}'>{value}</div>")
                    section.append("</div>")
                
                section.append("</div>") # Close summary-box
                
                # Add images if available
                image_paths = [
                    (f"{symbol}_{timeframe}_chart.png", "Price Chart with Indicators"),
                    (f"{symbol}_{timeframe}_trades_chart.png", "Trades"),
                    (f"{symbol}_{timeframe}_equity.png", "Equity Curve"),
                    (f"{symbol}_{timeframe}_equity_drawdown.png", "Drawdown")
                ]
                
                for img_path, title in image_paths:
                    full_path = os.path.join(self.results_dir, img_path)
                    if os.path.exists(full_path):
                        section.append("<div class='chart-container'>")
                        section.append(f"<h5>{title}</h5>")
                        if os.path.basename(self.output_path) == os.path.basename(self.results_dir) + ".html":
                            # Use relative path if report is in the same directory
                            section.append(f"<img src='{img_path}' style='width:100%;max-width:1000px;'>")
                        else:
                            # Use base64 encoding
                            with open(full_path, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            section.append(f"<img src='data:image/png;base64,{img_data}' style='width:100%;max-width:1000px;'>")
                        section.append("</div>")
                
                section.append("</div>") # Close timeframe tabcontent
            
            section.append("</div>") # Close symbol tabcontent
        
        return "\n".join(section)
    
    def _create_trades_analysis_section(self):
        """Create section with trades analysis."""
        section = []
        section.append("<h2>Trades Analysis</h2>")
        
        # Loop through symbols and timeframes
        for symbol, timeframe_results in self.results.items():
            for timeframe, _ in timeframe_results.items():
                # Load trades data
                trades_df = load_trade_data(self.results_dir, symbol, timeframe)
                
                if trades_df is not None and not trades_df.empty:
                    section.append(f"<h3>{symbol} - {timeframe}</h3>")
                    
                    # Analyze trades
                    trade_analysis = analyze_trades(trades_df)
                    
                    # Create metrics box
                    section.append("<div class='summary-box'>")
                    metrics = [
                        ("Total Trades", trade_analysis.get('total_trades', 0), "neutral"),
                        ("Win Rate", f"{trade_analysis.get('win_rate', 0) * 100:.2f}%", "good" if trade_analysis.get('win_rate', 0) >= 0.5 else "bad"),
                        ("Avg Win", f"{trade_analysis.get('avg_win', 0):.2f}%", "good"),
                        ("Avg Loss", f"{abs(trade_analysis.get('avg_loss', 0)):.2f}%", "bad"),
                        ("Max Consec. Wins", trade_analysis.get('max_consecutive_wins', 0), "good"),
                        ("Max Consec. Losses", trade_analysis.get('max_consecutive_losses', 0), "bad")
                    ]
                    
                    # Create metric boxes
                    for label, value, color in metrics:
                        section.append("<div class='metric-box'>")
                        section.append(f"<div class='metric-label'>{label}</div>")
                        section.append(f"<div class='metric-value {color}'>{value}</div>")
                        section.append("</div>")
                    
                    section.append("</div>") # Close summary-box
                    
                    # Create trades visualization
                    img_base64 = self._create_trades_plot(trades_df, symbol, timeframe)
                    if img_base64:
                        section.append("<div class='chart-container'>")
                        section.append("<h4>Trade Results Distribution</h4>")
                        section.append(f"<img src='data:image/png;base64,{img_base64}' style='width:100%;max-width:800px;'>")
                        section.append("</div>")
                    
                    # Display recent trades
                    if len(trades_df) > 0:
                        section.append("<h4>Recent Trades</h4>")
                        # Get relevant columns
                        recent_trades = trades_df.tail(10)[['signal', 'signal_reason', 'entry_price', 'exit_price', 'pnl']].copy()
                        # Add a strategy column
                        recent_trades['strategy'] = recent_trades['signal'].apply(lambda x: "Long" if x == 1 else "Short")
                        # Format columns for display
                        recent_trades['entry_price'] = recent_trades['entry_price'].map('${:.2f}'.format)
                        recent_trades['exit_price'] = recent_trades['exit_price'].map('${:.2f}'.format)
                        recent_trades['pnl'] = recent_trades['pnl'].map('{:.2f}%'.format)
                        # Rename columns
                        recent_trades.columns = ['Signal', 'Reason', 'Entry Price', 'Exit Price', 'PnL', 'Strategy']
                        # Reorder columns
                        recent_trades = recent_trades[['Strategy', 'Reason', 'Entry Price', 'Exit Price', 'PnL']]
                        # Convert to HTML
                        section.append(recent_trades.to_html(index=True))
        
        return "\n".join(section)
    
    def _create_comparison_plot(self, comparison_df):
        """Create a plot comparing performance across symbols and timeframes."""
        if comparison_df.empty:
            return None
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Create a grouped bar chart
            sns.set(style="whitegrid")
            
            # Create plot
            ax = sns.barplot(
                x="Symbol", 
                y="Total Return (%)", 
                hue="Timeframe", 
                data=comparison_df
            )
            
            # Customize plot
            plt.title("Total Return by Symbol and Timeframe", fontsize=16)
            plt.xlabel("Symbol", fontsize=12)
            plt.ylabel("Total Return (%)", fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(title="Timeframe", fontsize=10)
            
            # Add value labels on the bars
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    fontsize=8
                )
            
            # Save to BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
        
        except Exception as e:
            logger.error(f"Error creating comparison plot: {str(e)}")
            return None
    
    def _create_trades_plot(self, trades_df, symbol, timeframe):
        """Create a plot visualizing trade results."""
        if trades_df is None or trades_df.empty:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: PnL distribution
            sns.histplot(trades_df['pnl'], bins=20, kde=True, ax=ax1)
            ax1.axvline(x=0, color='r', linestyle='--')
            ax1.set_title(f"{symbol} - {timeframe} PnL Distribution", fontsize=14)
            ax1.set_xlabel("PnL (%)", fontsize=12)
            ax1.set_ylabel("Frequency", fontsize=12)
            
            # Plot 2: Cumulative PnL
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            ax2.plot(trades_df.index, trades_df['cumulative_pnl'])
            ax2.set_title(f"{symbol} - {timeframe} Cumulative PnL", fontsize=14)
            ax2.set_xlabel("Trade Number", fontsize=12)
            ax2.set_ylabel("Cumulative PnL (%)", fontsize=12)
            ax2.grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
        
        except Exception as e:
            logger.error(f"Error creating trades plot: {str(e)}")
            return None


def generate_report(results_dir, output_path=None):
    """
    Generate a comprehensive HTML report from backtest results.
    
    Parameters:
    -----------
    results_dir : str
        Path to the directory containing backtest results
    output_path : str
        Path to save the HTML report to
        
    Returns:
    --------
    str
        Path to the generated report
    """
    try:
        generator = ReportGenerator(results_dir, output_path)
        return generator.generate_report()
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None
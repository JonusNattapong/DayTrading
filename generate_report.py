"""
Command-line interface for generating reports from backtest results.
This script provides a simple way to generate HTML reports.
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.report_generator import generate_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("report_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function for the report generation script."""
    parser = argparse.ArgumentParser(description="Generate HTML reports from backtest results")
    parser.add_argument("--results-dir", required=True, help="Directory containing backtest results")
    parser.add_argument("--output", help="Path to save the HTML report")
    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_path = args.output
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Generate report
    logger.info(f"Generating report from {results_dir}")
    report_path = generate_report(results_dir, output_path)
    
    if report_path:
        logger.info(f"Report generated successfully: {report_path}")
        print(f"\nReport generated successfully: {report_path}")
    else:
        logger.error("Failed to generate report")
        print("\nFailed to generate report. Check the log for details.")


if __name__ == "__main__":
    main()
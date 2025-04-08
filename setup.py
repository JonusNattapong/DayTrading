#!/usr/bin/env python
"""
Setup script for the Day Trading Strategy project.
This script helps setting up the environment and dependencies.
"""
import os
import sys
import json
import argparse
import subprocess
import platform
from datetime import datetime

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print header text."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}==== {text} ===={Colors.ENDC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.WARNING}! {text}{Colors.ENDC}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}→ {text}{Colors.ENDC}")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    required_version = (3, 7)
    current_version = sys.version_info
    
    print_info(f"Current Python version: {sys.version}")
    print_info(f"Required Python version: {required_version[0]}.{required_version[1]} or higher")
    
    if current_version >= required_version:
        print_success("Python version is compatible")
        return True
    else:
        print_error(f"Python version {current_version[0]}.{current_version[1]} is not compatible")
        print_info("Please install Python 3.7 or higher")
        return False


def install_dependencies(args):
    """Install required dependencies."""
    print_header("Installing Dependencies")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print_error(f"Requirements file '{requirements_file}' not found")
        return False
    
    print_info(f"Installing packages from {requirements_file}")
    
    try:
        if args.virtual_env:
            # Create virtual environment
            venv_name = "day_trading_env"
            print_info(f"Creating virtual environment '{venv_name}'")
            
            subprocess.check_call([sys.executable, "-m", "venv", venv_name])
            
            # Determine activation script and pip based on platform
            if platform.system() == "Windows":
                pip_path = os.path.join(venv_name, "Scripts", "pip")
                activate_script = os.path.join(venv_name, "Scripts", "activate")
            else:
                pip_path = os.path.join(venv_name, "bin", "pip")
                activate_script = os.path.join(venv_name, "bin", "activate")
            
            # Install dependencies in the virtual environment
            subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
            subprocess.check_call([pip_path, "install", "-r", requirements_file])
            
            print_success(f"Virtual environment created at '{venv_name}'")
            print_info(f"To activate, run: {activate_script}")
        else:
            # Install in the current Python environment
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        
        print_success("Dependencies installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print_error(f"Error installing dependencies: {str(e)}")
        return False


def create_directories():
    """Create required directories if they don't exist."""
    print_header("Creating Directories")
    
    directories = ["data", "results"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print_success(f"Created directory: {directory}")
        else:
            print_info(f"Directory already exists: {directory}")
    
    return True


def check_configuration():
    """Check if configuration file exists and is valid."""
    print_header("Checking Configuration")
    
    config_path = "config/config.json"
    
    if not os.path.exists(config_path):
        print_error(f"Configuration file '{config_path}' not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for required sections
        required_sections = ["strategy", "backtest", "data", "output"]
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            print_error(f"Configuration file is missing required sections: {', '.join(missing_sections)}")
            return False
        
        print_success("Configuration file is valid")
        return True
    
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in configuration file: {str(e)}")
        return False
    except Exception as e:
        print_error(f"Error checking configuration: {str(e)}")
        return False


def setup_jupyter_notebook():
    """Set up Jupyter notebook if requested."""
    print_header("Setting Up Jupyter Notebook")
    
    try:
        # Check if jupyter is installed
        subprocess.check_call([sys.executable, "-m", "pip", "show", "jupyter"])
        print_success("Jupyter is already installed")
        
        # Check for nbextensions
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "show", "jupyter_contrib_nbextensions"])
            print_success("Jupyter notebook extensions are already installed")
        except subprocess.CalledProcessError:
            print_info("Installing Jupyter notebook extensions")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "jupyter_contrib_nbextensions"])
            subprocess.check_call([sys.executable, "-m", "jupyter", "contrib", "nbextension", "install", "--user"])
            print_success("Jupyter notebook extensions installed")
        
        print_info("To start Jupyter Notebook, run: jupyter notebook")
        return True
    
    except subprocess.CalledProcessError:
        print_warning("Jupyter is not installed")
        print_info("Installing Jupyter Notebook")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "jupyter", "jupyter_contrib_nbextensions"])
            subprocess.check_call([sys.executable, "-m", "jupyter", "contrib", "nbextension", "install", "--user"])
            print_success("Jupyter Notebook installed")
            print_info("To start Jupyter Notebook, run: jupyter notebook")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Error installing Jupyter: {str(e)}")
            return False


def display_summary():
    """Display setup summary and next steps."""
    print_header("Setup Summary")
    
    print(f"{Colors.BOLD}Day Trading Strategy Project Setup Complete{Colors.ENDC}")
    print("\nNext steps:")
    print("  1. Modify configuration in config/config.json")
    print("  2. Download market data or use cached data")
    print("  3. Run backtests with: python main.py")
    print("  4. Analyze results in the Jupyter notebook")
    
    print("\nUseful commands:")
    print("  - Run backtest: python main.py")
    print("  - Run backtest with custom config: python main.py --config path/to/config.json")
    print("  - Start Jupyter Notebook: jupyter notebook")
    
    print(f"\n{Colors.BOLD}Happy trading!{Colors.ENDC}")


def main():
    """Main function for the setup script."""
    parser = argparse.ArgumentParser(description="Setup script for Day Trading Strategy project")
    parser.add_argument("--virtual-env", action="store_true", help="Create and use a virtual environment")
    parser.add_argument("--skip-jupyter", action="store_true", help="Skip Jupyter notebook setup")
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}Day Trading Strategy Project Setup{Colors.ENDC}")
    print(f"Setup started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run setup steps
    if not check_python_version():
        sys.exit(1)
    
    if not install_dependencies(args):
        print_warning("Failed to install dependencies")
    
    if not create_directories():
        print_warning("Failed to create directories")
    
    if not check_configuration():
        print_warning("Failed to validate configuration")
    
    if not args.skip_jupyter:
        if not setup_jupyter_notebook():
            print_warning("Failed to set up Jupyter notebook")
    
    # Display summary
    display_summary()


if __name__ == "__main__":
    main()
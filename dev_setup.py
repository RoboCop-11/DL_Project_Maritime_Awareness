#!/usr/bin/env python3
"""
Development setup script for Maritime Domain Awareness System

This script helps set up the development environment and run common tasks.
"""

import subprocess
import sys
import os
import argparse

def run_command(command, description):
    """Run a command and print the result"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def setup_environment():
    """Set up development environment"""
    print("🚀 Setting up Maritime Domain Awareness System development environment")
    
    commands = [
        ("pip install -e .", "Installing package in development mode"),
        ("pip install -r requirements-dev.txt", "Installing development dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    print("✅ Development environment setup complete!")
    return True

def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite...")
    
    if not os.path.exists("tests"):
        print("❌ Tests directory not found")
        return False
    
    return run_command("python -m pytest tests/ -v", "Running tests")

def run_linting():
    """Run code quality checks"""
    print("🔍 Running code quality checks...")
    
    commands = [
        ("black --check src/ examples/ scripts/", "Checking code formatting with Black"),
        ("flake8 src/ examples/ scripts/", "Running Flake8 linter"),
        ("mypy src/", "Running MyPy type checking"),
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed

def format_code():
    """Format code using Black and isort"""
    print("🎨 Formatting code...")
    
    commands = [
        ("black src/ examples/ scripts/", "Formatting code with Black"),
        ("isort src/ examples/ scripts/", "Sorting imports with isort"),
    ]
    
    for command, description in commands:
        run_command(command, description)

def create_data_directories():
    """Create necessary data directories"""
    print("📁 Creating data directories...")
    
    directories = [
        "data/SSDD_coco",
        "models", 
        "output",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Development setup and utilities")
    parser.add_argument("--setup", action="store_true", help="Set up development environment")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--lint", action="store_true", help="Run code quality checks")
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--dirs", action="store_true", help="Create data directories")
    parser.add_argument("--all", action="store_true", help="Run setup, create dirs, format, lint, and test")
    
    args = parser.parse_args()
    
    if args.all:
        setup_environment()
        create_data_directories()
        format_code()
        run_linting()
        run_tests()
    else:
        if args.setup:
            setup_environment()
        if args.dirs:
            create_data_directories()
        if args.format:
            format_code()
        if args.lint:
            run_linting()
        if args.test:
            run_tests()
        
        if not any([args.setup, args.test, args.lint, args.format, args.dirs]):
            print("📋 Available commands:")
            print("  --setup    Set up development environment")
            print("  --dirs     Create necessary data directories")
            print("  --format   Format code with Black and isort")
            print("  --lint     Run code quality checks")
            print("  --test     Run test suite")
            print("  --all      Run all of the above")
            print("\nExample: python dev_setup.py --setup --dirs")

if __name__ == "__main__":
    main()
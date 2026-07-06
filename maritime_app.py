#!/usr/bin/env python3
"""
Maritime Domain Awareness System - Main Application Entry Point

This is the main entry point for running the Maritime Domain Awareness System.
It provides a simple interface to launch the dashboard or run examples.
"""

import sys
import os
import subprocess
import argparse

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Maritime Domain Awareness Dashboard...")
    print("📱 The dashboard will open in your web browser")
    print("🔗 URL: http://localhost:8501")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            os.path.join("src", "dashboard", "maritime_dashboard.py"),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def run_example(example_name):
    """Run an example script"""
    example_files = {
        'simple': 'examples/simple_maritime_demo.py',
        'complete': 'examples/complete_demo.py', 
        'tracking': 'examples/run_maritime_tracking.py'
    }
    
    if example_name not in example_files:
        print(f"❌ Unknown example: {example_name}")
        print(f"Available examples: {', '.join(example_files.keys())}")
        return
    
    example_file = example_files[example_name]
    if not os.path.exists(example_file):
        print(f"❌ Example file not found: {example_file}")
        return
    
    print(f"🔄 Running example: {example_name}")
    try:
        subprocess.run([sys.executable, example_file])
    except Exception as e:
        print(f"❌ Error running example: {e}")

def setup_environment():
    """Set up the environment"""
    print("🔧 Setting up Maritime Domain Awareness System...")
    
    # Check if required files exist
    required_files = [
        "src/dashboard/maritime_dashboard.py",
        "src/core/maritime_tracking_system.py",
        "models/best.pt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        if "models/best.pt" in missing_files:
            print("\nℹ️ Please place your trained YOLO model as 'models/best.pt'")
        return False
    
    # Try to install dependencies if not available
    try:
        import streamlit
        import cv2
        import numpy as np
        import torch
        import ultralytics
    except ImportError as e:
        print(f"⚠️ Missing dependency: {e}")
        print("📦 Installing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "streamlit", "opencv-python", "torch", "ultralytics", "plotly", "seaborn"
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run:")
            print("   pip install -r requirements.txt")
            return False
    
    print("✅ Environment setup complete!")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Maritime Domain Awareness System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python maritime_app.py dashboard          # Launch web dashboard
  python maritime_app.py example simple     # Run simple demo
  python maritime_app.py example complete   # Run complete demo
  python maritime_app.py setup              # Set up environment
        """
    )
    
    parser.add_argument('command', choices=['dashboard', 'example', 'setup'],
                       help='Command to run')
    parser.add_argument('subcommand', nargs='?', 
                       help='Subcommand (for example: simple, complete, tracking)')
    
    args = parser.parse_args()
    
    print("🚢 Maritime Domain Awareness System")
    print("=" * 50)
    
    if args.command == 'setup':
        setup_environment()
    elif args.command == 'dashboard':
        if setup_environment():
            launch_dashboard()
    elif args.command == 'example':
        if not args.subcommand:
            print("❌ Please specify an example: simple, complete, or tracking")
            return
        if setup_environment():
            run_example(args.subcommand)

if __name__ == "__main__":
    main()
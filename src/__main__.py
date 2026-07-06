"""
Entry point for running the maritime tracking system as a module
"""

import sys
import os

def main():
    """Main entry point"""
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    
    from src.dashboard.maritime_dashboard import main as dashboard_main
    dashboard_main()

if __name__ == "__main__":
    main()
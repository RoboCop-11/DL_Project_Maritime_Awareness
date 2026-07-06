"""
Maritime Dashboard Launcher
Simple script to launch the Streamlit dashboard
"""

import subprocess
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit and dashboard requirements"""
    print("📦 Installing Streamlit and dashboard requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "seaborn"])
        print("✅ Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Streamlit: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Maritime Domain Awareness Dashboard...")
    print("📱 The dashboard will open in your web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    print("🚢 MARITIME DOMAIN AWARENESS SYSTEM")
    print("="*50)
    print("Features:")
    print("• 📤 Upload SAR images")
    print("• 🔍 Real-time ship detection")
    print("• 🛤️ Multi-object tracking")
    print("• 🚨 Anomaly detection")
    print("• 📊 Interactive visualizations")
    print("• 📋 Comprehensive reports")
    print("="*50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/dashboard/maritime_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print("🚢 Maritime Domain Awareness System - Dashboard Launcher")
    print("=" * 60)
    
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
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all system files are present before launching the dashboard.")
        return
    
    # Check Streamlit installation
    if not check_streamlit():
        print("⚠️ Streamlit not found. Installing...")
        if not install_streamlit():
            print("❌ Failed to install Streamlit. Please install manually:")
            print("   pip install streamlit plotly seaborn")
            return
    
    print("✅ All requirements satisfied!")
    print("\n🎯 Ready to launch dashboard...")
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()
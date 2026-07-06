"""
Simple test to verify dashboard components work
"""

import sys
import os
import numpy as np

def test_dashboard_imports():
    """Test dashboard imports"""
    print("🧪 Testing Dashboard Imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
        
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported")
        
        from maritime_tracking_system import MaritimeTrackingSystem
        print("✅ Maritime system imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_trajectory_plot_function():
    """Test the trajectory plot function with sample data"""
    print("\n🧪 Testing Trajectory Plot Function...")
    
    try:
        # Import the function
        sys.path.append('.')
        from maritime_dashboard import create_trajectory_plot
        
        # Create sample track data
        sample_data = [
            {'track_id': 0, 'frame_id': 0, 'position': [100, 100]},
            {'track_id': 0, 'frame_id': 1, 'position': [105, 102]},
            {'track_id': 0, 'frame_id': 2, 'position': [110, 104]},
            {'track_id': 1, 'frame_id': 0, 'position': [200, 150]},
            {'track_id': 1, 'frame_id': 1, 'position': [205, 152]},
        ]
        
        # Test the function
        fig = create_trajectory_plot(sample_data)
        
        if fig is not None:
            print("✅ Trajectory plot function works")
            return True
        else:
            print("⚠️ Trajectory plot returned None")
            return False
            
    except Exception as e:
        print(f"❌ Trajectory plot test failed: {e}")
        return False

def main():
    """Run simple dashboard tests"""
    print("🚢 Simple Dashboard Test")
    print("=" * 30)
    
    tests_passed = 0
    total_tests = 2
    
    if test_dashboard_imports():
        tests_passed += 1
    
    if test_trajectory_plot_function():
        tests_passed += 1
    
    print(f"\n📊 Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ Dashboard components working!")
        print("\n🚀 Ready to launch dashboard:")
        print("   streamlit run maritime_dashboard.py")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
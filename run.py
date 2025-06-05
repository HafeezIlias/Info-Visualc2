#!/usr/bin/env python3
"""
CO2 Emissions Analysis Web Application Runner

This script checks for required files and dependencies, then starts the Flask application.
"""

import os
import sys
import subprocess

def check_file_exists(filename):
    """Check if a file exists in the current directory."""
    return os.path.isfile(filename)

def check_dependencies():
    """Check if required Python packages are installed."""
    required_packages = ['flask', 'pandas', 'numpy', 'plotly', 'sklearn']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def main():
    """Main function to run the application."""
    print("🌍 CO2 Emissions Analysis Web Application")
    print("=" * 50)
    
    # Check for required files
    required_files = [
        'app.py',
        'Co2_Emissions_by_Sectors_Europe-Asia.csv',
        'templates/base.html',
        'templates/index.html',
        'templates/visualization.html',
        'templates/prediction.html'
    ]
    
    missing_files = [f for f in required_files if not check_file_exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all files are in the correct locations.")
        return 1
    
    # Check for model files
    model_files = ['best_model.pkl', 'scaler.pkl', 'model_info.pkl']
    missing_models = [f for f in model_files if not check_file_exists(f)]
    
    if missing_models:
        print("⚠️  Missing model files:")
        for file in missing_models:
            print(f"   - {file}")
        print("\n🔧 Generating model files...")
        try:
            subprocess.run([sys.executable, 'model.py'], check=True)
            print("✅ Model files generated successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to generate model files. Please run 'python model.py' manually.")
            return 1
        except FileNotFoundError:
            print("❌ model.py not found. Please ensure it exists in the current directory.")
            return 1
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("❌ Missing required packages:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Install missing packages with: pip install -r requirements.txt")
        return 1
    
    print("✅ All checks passed!")
    print("\n🚀 Starting the web application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask application
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 
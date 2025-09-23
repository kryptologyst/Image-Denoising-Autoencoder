#!/usr/bin/env python3
"""
Setup Script for Image Denoising Autoencoder
============================================

This script sets up the project environment and creates necessary directories.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'logs', 'data', 'data/synthetic', 'data/real_world', 'data/medical', 'data/satellite']
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✓ {directory}")

def check_dependencies():
    """Check if required packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'tensorflow', 'numpy', 'matplotlib', 'opencv-python', 
        'pillow', 'streamlit', 'scikit-image', 'tqdm', 'seaborn', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def create_sample_data():
    """Create sample datasets"""
    print("\n🗄️ Creating sample datasets...")
    
    try:
        from mock_database import create_sample_datasets
        create_sample_datasets()
        print("   ✓ Sample datasets created")
    except Exception as e:
        print(f"   ✗ Failed to create sample datasets: {e}")

def main():
    """Main setup function"""
    print("🚀 Image Denoising Autoencoder Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup incomplete - missing dependencies")
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run demo: python demo.py")
    print("2. Start web UI: streamlit run streamlit_app.py")
    print("3. Train models: python advanced_autoencoder.py")

if __name__ == "__main__":
    main()

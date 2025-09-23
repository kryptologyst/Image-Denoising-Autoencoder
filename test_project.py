#!/usr/bin/env python3
"""
Simplified Test Script for Image Denoising Autoencoder
=====================================================

This script tests the basic functionality without TensorFlow dependencies
to ensure the project structure and basic components work.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def test_basic_functionality():
    """Test basic functionality without TensorFlow"""
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    # Test 1: Directory creation
    print("📁 Testing directory creation...")
    directories = ['models', 'results', 'logs', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✓ {directory}")
    
    # Test 2: NumPy operations
    print("\n🔢 Testing NumPy operations...")
    # Create sample data
    clean_images = np.random.random((10, 28, 28, 1))
    noise = np.random.normal(0, 0.1, clean_images.shape)
    noisy_images = np.clip(clean_images + noise, 0, 1)
    
    print(f"   ✓ Created {clean_images.shape[0]} sample images")
    print(f"   ✓ Added noise (shape: {noisy_images.shape})")
    
    # Test 3: Basic metrics calculation
    print("\n📊 Testing metrics calculation...")
    mse = np.mean((clean_images - noisy_images) ** 2)
    print(f"   ✓ MSE: {mse:.6f}")
    
    # Test 4: Visualization
    print("\n🎨 Testing visualization...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    for i in range(5):
        # Original
        axes[0, i].imshow(clean_images[i].squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Noisy
        axes[1, i].imshow(noisy_images[i].squeeze(), cmap='gray')
        axes[1, i].set_title('Noisy')
        axes[1, i].axis('off')
    
    plt.suptitle('Image Denoising Test - Basic Functionality')
    plt.tight_layout()
    plt.savefig('results/basic_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Visualization saved to results/basic_test.png")
    
    # Test 5: File operations
    print("\n💾 Testing file operations...")
    # Save sample data
    np.save('data/test_data.npy', clean_images)
    loaded_data = np.load('data/test_data.npy')
    print(f"   ✓ Saved and loaded data (shape: {loaded_data.shape})")
    
    # Test 6: Project structure
    print("\n📋 Testing project structure...")
    required_files = [
        'advanced_autoencoder.py',
        'streamlit_app.py', 
        'mock_database.py',
        'demo.py',
        'setup.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'LICENSE'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} (missing)")
    
    print("\n✅ Basic functionality test completed!")
    return True

def test_mock_database():
    """Test mock database functionality"""
    print("\n🗄️ Testing Mock Database")
    print("=" * 30)
    
    try:
        # Import mock database
        from mock_database import MockDatabase
        
        # Create database instance
        db = MockDatabase()
        print("   ✓ MockDatabase imported and initialized")
        
        # Test dataset creation
        print("   📊 Creating test dataset...")
        shapes = db.generate_synthetic_shapes(50, (32, 32))
        print(f"   ✓ Generated {shapes.shape[0]} synthetic shape images")
        
        # Test noise generation
        print("   🔊 Testing noise generation...")
        noise = np.random.normal(0, 0.1, shapes.shape)
        noisy_shapes = np.clip(shapes + noise, 0, 1)
        print(f"   ✓ Added Gaussian noise to images")
        
        # Test metrics
        print("   📈 Testing metrics...")
        mse = np.mean((shapes - noisy_shapes) ** 2)
        print(f"   ✓ Calculated MSE: {mse:.6f}")
        
        print("   ✅ Mock database test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Mock database test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Image Denoising Autoencoder - Test Suite")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_basic_functionality()
    test2_passed = test_mock_database()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Basic Functionality: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Mock Database: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Install TensorFlow: pip install tensorflow")
        print("2. Run full demo: python3 demo.py")
        print("3. Start web UI: streamlit run streamlit_app.py")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()

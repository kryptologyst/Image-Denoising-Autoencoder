#!/usr/bin/env python3
"""
Quick Demo Script for Image Denoising Autoencoder
================================================

This script provides a quick demonstration of the autoencoder capabilities
without requiring the full training process.
"""

import numpy as np
import matplotlib.pyplot as plt
from advanced_autoencoder import ImageDenoisingAutoencoder, DatasetManager
from mock_database import MockDatabase

def quick_demo():
    """Run a quick demonstration"""
    print("🚀 Image Denoising Autoencoder - Quick Demo")
    print("=" * 50)
    
    # Load MNIST dataset
    print("📥 Loading MNIST dataset...")
    x_train, x_test, _, _ = DatasetManager.load_mnist()
    print(f"   Training samples: {x_train.shape[0]}")
    print(f"   Test samples: {x_test.shape[0]}")
    print(f"   Image shape: {x_train.shape[1:]}")
    
    # Initialize autoencoder
    print("\n🏗️ Initializing U-Net autoencoder...")
    autoencoder = ImageDenoisingAutoencoder(
        input_shape=(28, 28, 1),
        noise_factor=0.5,
        model_type='unet'
    )
    
    # Build model
    model = autoencoder.build_model()
    autoencoder.compile_model()
    
    print(f"   Model parameters: {model.count_params():,}")
    print(f"   Model summary:")
    model.summary()
    
    # Quick training (just a few epochs for demo)
    print("\n🏋️ Training model (5 epochs for demo)...")
    history = autoencoder.train(
        x_train[:1000],  # Use subset for speed
        x_test[:200],    # Use subset for speed
        epochs=5,
        batch_size=64
    )
    
    # Test on sample images
    print("\n🎨 Testing on sample images...")
    sample_images = x_test[:10]
    noisy_images = autoencoder.add_noise(sample_images)
    denoised_images = autoencoder.predict(noisy_images)
    
    # Calculate metrics
    metrics = autoencoder.evaluate_metrics(sample_images, denoised_images)
    print(f"   PSNR: {metrics['PSNR']:.2f} dB")
    print(f"   SSIM: {metrics['SSIM']:.4f}")
    print(f"   MSE: {metrics['MSE']:.6f}")
    
    # Visualize results
    print("\n📊 Visualizing results...")
    autoencoder.visualize_results(
        sample_images, noisy_images, denoised_images,
        n_samples=5,
        save_path='demo_results.png'
    )
    
    print("\n✅ Demo completed successfully!")
    print("   Check 'demo_results.png' for visualization")

def mock_database_demo():
    """Demonstrate mock database capabilities"""
    print("\n🗄️ Mock Database Demo")
    print("=" * 30)
    
    db = MockDatabase()
    
    # Create sample datasets
    print("Creating sample datasets...")
    
    # Geometric shapes
    db.create_dataset("demo_shapes", "synthetic", 100, image_size=(32, 32))
    print("✓ Created geometric shapes dataset")
    
    # Text images
    db.create_dataset("demo_text", "synthetic", 50, image_size=(32, 32))
    print("✓ Created text images dataset")
    
    # List available datasets
    print("\nAvailable datasets:")
    for dataset_info in db.list_datasets():
        print(f"  - {dataset_info['name']}: {dataset_info['n_samples']} samples")
    
    # Visualize a dataset
    print("\nVisualizing geometric shapes dataset...")
    db.visualize_dataset("demo_shapes", n_samples=8)

if __name__ == "__main__":
    try:
        quick_demo()
        mock_database_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

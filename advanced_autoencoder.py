"""
Project 111: Advanced Image Denoising Autoencoder
=================================================

A modern implementation of image denoising using various autoencoder architectures
with support for multiple datasets, evaluation metrics, and interactive UI.

Features:
- Multiple autoencoder architectures (Basic CNN, U-Net, ResNet-based)
- Support for MNIST, CIFAR-10, and custom datasets
- Advanced evaluation metrics (PSNR, SSIM, MSE)
- Model saving/loading functionality
- Interactive Streamlit UI
- Modern TensorFlow/Keras practices
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    Conv2DTranspose, BatchNormalization, Dropout,
    Add, Concatenate, Activation, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.utils import plot_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class ImageDenoisingAutoencoder:
    """
    Advanced Image Denoising Autoencoder with multiple architectures
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (28, 28, 1), 
                 noise_factor: float = 0.5, model_type: str = 'unet'):
        """
        Initialize the autoencoder
        
        Args:
            input_shape: Input image shape (height, width, channels)
            noise_factor: Amount of noise to add (0.0 to 1.0)
            model_type: Type of model ('basic', 'unet', 'resnet')
        """
        self.input_shape = input_shape
        self.noise_factor = noise_factor
        self.model_type = model_type
        self.model = None
        self.history = None
        
        # Create directories
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = ['models', 'results', 'logs', 'data']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def add_noise(self, images: np.ndarray, noise_type: str = 'gaussian') -> np.ndarray:
        """
        Add noise to images
        
        Args:
            images: Clean images
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'poisson')
            
        Returns:
            Noisy images
        """
        if noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_factor, images.shape)
            noisy_images = images + noise
        elif noise_type == 'salt_pepper':
            noisy_images = images.copy()
            # Salt noise
            salt_mask = np.random.random(images.shape) < self.noise_factor / 2
            noisy_images[salt_mask] = 1
            # Pepper noise
            pepper_mask = np.random.random(images.shape) < self.noise_factor / 2
            noisy_images[pepper_mask] = 0
        elif noise_type == 'poisson':
            noisy_images = np.random.poisson(images * 255) / 255.0
        
        return np.clip(noisy_images, 0., 1.)
    
    def build_basic_autoencoder(self) -> Model:
        """Build basic convolutional autoencoder"""
        input_img = Input(shape=self.input_shape)
        
        # Encoder
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(self.input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(input_img, decoded)
        return autoencoder
    
    def build_unet_autoencoder(self) -> Model:
        """Build U-Net style autoencoder with skip connections"""
        input_img = Input(shape=self.input_shape)
        
        # Encoder
        e1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        e1 = BatchNormalization()(e1)
        e1 = Conv2D(64, (3, 3), activation='relu', padding='same')(e1)
        p1 = MaxPooling2D((2, 2))(e1)
        
        e2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        e2 = BatchNormalization()(e2)
        e2 = Conv2D(128, (3, 3), activation='relu', padding='same')(e2)
        p2 = MaxPooling2D((2, 2))(e2)
        
        e3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        e3 = BatchNormalization()(e3)
        e3 = Conv2D(256, (3, 3), activation='relu', padding='same')(e3)
        
        # Decoder with skip connections
        d1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(e3)
        d1 = Concatenate()([d1, e2])
        d1 = Conv2D(128, (3, 3), activation='relu', padding='same')(d1)
        d1 = BatchNormalization()(d1)
        d1 = Conv2D(128, (3, 3), activation='relu', padding='same')(d1)
        
        d2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
        d2 = Concatenate()([d2, e1])
        d2 = Conv2D(64, (3, 3), activation='relu', padding='same')(d2)
        d2 = BatchNormalization()(d2)
        d2 = Conv2D(64, (3, 3), activation='relu', padding='same')(d2)
        
        decoded = Conv2D(self.input_shape[-1], (1, 1), activation='sigmoid')(d2)
        
        autoencoder = Model(input_img, decoded)
        return autoencoder
    
    def build_resnet_autoencoder(self) -> Model:
        """Build ResNet-based autoencoder"""
        input_img = Input(shape=self.input_shape)
        
        # Encoder with ResNet blocks
        x = Conv2D(64, (7, 7), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        # ResNet blocks
        for _ in range(3):
            residual = x
            x = Conv2D(64, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Conv2D(64, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Add()([x, residual])
            x = LeakyReLU()(x)
        
        x = MaxPooling2D((2, 2))(x)
        
        # Decoder
        x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        
        decoded = Conv2D(self.input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(input_img, decoded)
        return autoencoder
    
    def build_model(self) -> Model:
        """Build the specified model architecture"""
        if self.model_type == 'basic':
            self.model = self.build_basic_autoencoder()
        elif self.model_type == 'unet':
            self.model = self.build_unet_autoencoder()
        elif self.model_type == 'resnet':
            self.model = self.build_resnet_autoencoder()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with optimizer and loss function"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, x_train: np.ndarray, x_test: np.ndarray, 
              epochs: int = 50, batch_size: int = 32, 
              validation_split: float = 0.1) -> Dict[str, Any]:
        """
        Train the autoencoder
        
        Args:
            x_train: Training images
            x_test: Test images
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            Training history
        """
        # Add noise to training data
        x_train_noisy = self.add_noise(x_train)
        x_test_noisy = self.add_noise(x_test)
        
        # Build and compile model
        self.build_model()
        self.compile_model()
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                f'models/best_{self.model_type}_autoencoder.h5',
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            TensorBoard(
                log_dir=f'logs/{self.model_type}',
                histogram_freq=1
            )
        ]
        
        # Train the model
        logger.info(f"Training {self.model_type} autoencoder...")
        self.history = self.model.fit(
            x_train_noisy, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss = self.model.evaluate(x_test_noisy, x_test, verbose=0)
        logger.info(f"Test loss: {test_loss[0]:.4f}, Test MAE: {test_loss[1]:.4f}")
        
        return self.history.history
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict denoised images"""
        return self.model.predict(images, verbose=0)
    
    def evaluate_metrics(self, original: np.ndarray, denoised: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            original: Original clean images
            denoised: Denoised images
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # MSE
        mse = mean_squared_error(original.flatten(), denoised.flatten())
        metrics['MSE'] = mse
        
        # PSNR
        if mse > 0:
            psnr = peak_signal_noise_ratio(original, denoised, data_range=1.0)
            metrics['PSNR'] = psnr
        
        # SSIM
        ssim_values = []
        for i in range(len(original)):
            ssim_val = structural_similarity(
                original[i].squeeze(), 
                denoised[i].squeeze(),
                data_range=1.0
            )
            ssim_values.append(ssim_val)
        metrics['SSIM'] = np.mean(ssim_values)
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def visualize_results(self, original: np.ndarray, noisy: np.ndarray, 
                         denoised: np.ndarray, n_samples: int = 10, 
                         save_path: Optional[str] = None):
        """
        Visualize denoising results
        
        Args:
            original: Original clean images
            noisy: Noisy images
            denoised: Denoised images
            n_samples: Number of samples to display
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(3, n_samples, figsize=(20, 6))
        
        for i in range(n_samples):
            # Original
            axes[0, i].imshow(original[i].squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Noisy
            axes[1, i].imshow(noisy[i].squeeze(), cmap='gray')
            axes[1, i].set_title('Noisy')
            axes[1, i].axis('off')
            
            # Denoised
            axes[2, i].imshow(denoised[i].squeeze(), cmap='gray')
            axes[2, i].set_title('Denoised')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        
        plt.show()


class DatasetManager:
    """Manager for different datasets"""
    
    @staticmethod
    def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MNIST dataset"""
        (x_train, _), (x_test, _) = mnist.load_data()
        
        # Normalize and reshape
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        return x_train, x_test, None, None
    
    @staticmethod
    def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load CIFAR-10 dataset"""
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Normalize
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        return x_train, x_test, y_train, y_test
    
    @staticmethod
    def create_custom_dataset(image_dir: str, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """Create custom dataset from directory of images"""
        images = []
        
        for img_path in Path(image_dir).glob('*.jpg'):
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
        
        return np.array(images)


def main():
    """Main function to demonstrate the autoencoder"""
    logger.info("Starting Image Denoising Autoencoder Demo")
    
    # Load MNIST dataset
    x_train, x_test, _, _ = DatasetManager.load_mnist()
    
    # Train different models
    models = ['basic', 'unet', 'resnet']
    results = {}
    
    for model_type in models:
        logger.info(f"\nTraining {model_type} autoencoder...")
        
        # Initialize autoencoder
        autoencoder = ImageDenoisingAutoencoder(
            input_shape=(28, 28, 1),
            noise_factor=0.5,
            model_type=model_type
        )
        
        # Train model
        history = autoencoder.train(
            x_train, x_test,
            epochs=20,  # Reduced for demo
            batch_size=128
        )
        
        # Evaluate
        x_test_noisy = autoencoder.add_noise(x_test)
        denoised = autoencoder.predict(x_test_noisy)
        
        metrics = autoencoder.evaluate_metrics(x_test, denoised)
        results[model_type] = metrics
        
        # Visualize results
        autoencoder.visualize_results(
            x_test[:10], x_test_noisy[:10], denoised[:10],
            save_path=f'results/{model_type}_results.png'
        )
        
        # Plot training history
        autoencoder.plot_training_history(
            save_path=f'results/{model_type}_history.png'
        )
        
        # Save model
        autoencoder.save_model(f'models/{model_type}_autoencoder.h5')
    
    # Compare results
    logger.info("\nModel Comparison:")
    for model_type, metrics in results.items():
        logger.info(f"{model_type.upper()}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()

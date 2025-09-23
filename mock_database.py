"""
Mock Database for Image Denoising Autoencoder
==============================================

This module provides a mock database with various datasets and utilities
for generating synthetic data for testing and demonstration purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
from sklearn.datasets import make_blobs
import random

class MockDatabase:
    """Mock database for storing and retrieving image datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "synthetic").mkdir(exist_ok=True)
        (self.data_dir / "real_world").mkdir(exist_ok=True)
        (self.data_dir / "medical").mkdir(exist_ok=True)
        (self.data_dir / "satellite").mkdir(exist_ok=True)
        
        self.datasets = {}
        self.load_existing_datasets()
    
    def load_existing_datasets(self):
        """Load existing datasets from disk"""
        metadata_file = self.data_dir / "datasets_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.datasets = json.load(f)
    
    def save_metadata(self):
        """Save dataset metadata to disk"""
        metadata_file = self.data_dir / "datasets_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.datasets, f, indent=2)
    
    def generate_synthetic_shapes(self, n_samples: int = 1000, 
                                image_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """Generate synthetic images with geometric shapes"""
        images = []
        
        for _ in range(n_samples):
            # Create blank image
            img = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(img)
            
            # Random number of shapes
            num_shapes = random.randint(1, 5)
            
            for _ in range(num_shapes):
                # Random shape type
                shape_type = random.choice(['circle', 'rectangle', 'triangle', 'line'])
                
                # Random position and size
                x1 = random.randint(0, image_size[0]//2)
                y1 = random.randint(0, image_size[1]//2)
                x2 = random.randint(image_size[0]//2, image_size[0])
                y2 = random.randint(image_size[1]//2, image_size[1])
                
                # Random intensity
                intensity = random.randint(100, 255)
                
                if shape_type == 'circle':
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    radius = min(x2 - x1, y2 - y1) // 2
                    draw.ellipse([center[0] - radius, center[1] - radius,
                                 center[0] + radius, center[1] + radius],
                                fill=intensity)
                elif shape_type == 'rectangle':
                    draw.rectangle([x1, y1, x2, y2], fill=intensity)
                elif shape_type == 'triangle':
                    points = [(x1, y2), (x2, y2), ((x1 + x2) // 2, y1)]
                    draw.polygon(points, fill=intensity)
                elif shape_type == 'line':
                    draw.line([x1, y1, x2, y2], fill=intensity, width=random.randint(2, 5))
            
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            images.append(img_array)
        
        return np.array(images)
    
    def generate_text_images(self, n_samples: int = 500, 
                           image_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """Generate synthetic images with text"""
        images = []
        
        # Sample text strings
        text_samples = [
            "HELLO", "WORLD", "AI", "ML", "DEEP", "LEARNING",
            "TENSOR", "FLOW", "PYTHON", "CODE", "DATA", "SCIENCE"
        ]
        
        for _ in range(n_samples):
            # Create blank image
            img = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(img)
            
            # Random text
            text = random.choice(text_samples)
            
            # Try to use a font, fallback to default
            try:
                font_size = random.randint(12, 24)
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Random position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = random.randint(0, max(1, image_size[0] - text_width))
            y = random.randint(0, max(1, image_size[1] - text_height))
            
            # Random intensity
            intensity = random.randint(150, 255)
            
            draw.text((x, y), text, fill=intensity, font=font)
            
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            images.append(img_array)
        
        return np.array(images)
    
    def generate_medical_images(self, n_samples: int = 300, 
                               image_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Generate synthetic medical images (X-ray like)"""
        images = []
        
        for _ in range(n_samples):
            # Create base image with gradient
            img = np.zeros(image_size)
            
            # Add anatomical structures
            # Spine
            spine_width = random.randint(3, 8)
            spine_x = image_size[1] // 2
            img[:, spine_x-spine_width:spine_x+spine_width] = 0.8
            
            # Ribs
            for i in range(random.randint(8, 12)):
                y = random.randint(0, image_size[0])
                x1 = random.randint(0, spine_x - 10)
                x2 = random.randint(spine_x + 10, image_size[1])
                img[y:y+2, x1:x1+random.randint(5, 15)] = 0.6
                img[y:y+2, x2:x2+random.randint(5, 15)] = 0.6
            
            # Organs (circles)
            for _ in range(random.randint(2, 4)):
                center_x = random.randint(20, image_size[1] - 20)
                center_y = random.randint(20, image_size[0] - 20)
                radius = random.randint(8, 20)
                
                y, x = np.ogrid[:image_size[0], :image_size[1]]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                img[mask] = random.uniform(0.3, 0.7)
            
            # Add noise
            noise = np.random.normal(0, 0.1, image_size)
            img = np.clip(img + noise, 0, 1)
            
            images.append(img)
        
        return np.array(images)
    
    def generate_satellite_images(self, n_samples: int = 400, 
                                 image_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """Generate synthetic satellite images"""
        images = []
        
        for _ in range(n_samples):
            # Create base terrain
            img = np.random.uniform(0.1, 0.4, image_size)
            
            # Add water bodies
            water_mask = np.random.random(image_size) < 0.2
            img[water_mask] = np.random.uniform(0.0, 0.2)
            
            # Add vegetation
            vegetation_mask = np.random.random(image_size) < 0.3
            img[vegetation_mask] = np.random.uniform(0.3, 0.6)
            
            # Add urban areas
            urban_mask = np.random.random(image_size) < 0.1
            img[urban_mask] = np.random.uniform(0.6, 0.9)
            
            # Add roads
            for _ in range(random.randint(2, 5)):
                if random.random() < 0.5:
                    # Horizontal road
                    y = random.randint(0, image_size[0] - 3)
                    img[y:y+3, :] = np.random.uniform(0.7, 0.9)
                else:
                    # Vertical road
                    x = random.randint(0, image_size[1] - 3)
                    img[:, x:x+3] = np.random.uniform(0.7, 0.9)
            
            images.append(img)
        
        return np.array(images)
    
    def generate_noise_patterns(self, clean_images: np.ndarray, 
                               noise_types: List[str] = None) -> Dict[str, np.ndarray]:
        """Generate different types of noise patterns"""
        if noise_types is None:
            noise_types = ['gaussian', 'salt_pepper', 'poisson', 'speckle']
        
        noisy_datasets = {}
        
        for noise_type in noise_types:
            noisy_images = clean_images.copy()
            
            if noise_type == 'gaussian':
                noise = np.random.normal(0, 0.1, clean_images.shape)
                noisy_images = np.clip(clean_images + noise, 0, 1)
            
            elif noise_type == 'salt_pepper':
                # Salt noise
                salt_mask = np.random.random(clean_images.shape) < 0.05
                noisy_images[salt_mask] = 1
                # Pepper noise
                pepper_mask = np.random.random(clean_images.shape) < 0.05
                noisy_images[pepper_mask] = 0
            
            elif noise_type == 'poisson':
                noisy_images = np.random.poisson(clean_images * 255) / 255.0
            
            elif noise_type == 'speckle':
                noise = np.random.normal(0, 0.1, clean_images.shape)
                noisy_images = clean_images + clean_images * noise
                noisy_images = np.clip(noisy_images, 0, 1)
            
            noisy_datasets[noise_type] = noisy_images
        
        return noisy_datasets
    
    def create_dataset(self, dataset_name: str, dataset_type: str, 
                      n_samples: int = 1000, **kwargs) -> str:
        """Create a new dataset"""
        dataset_path = self.data_dir / dataset_type / f"{dataset_name}.npy"
        
        if dataset_type == "synthetic":
            if "shapes" in dataset_name.lower():
                images = self.generate_synthetic_shapes(n_samples, **kwargs)
            elif "text" in dataset_name.lower():
                images = self.generate_text_images(n_samples, **kwargs)
            else:
                images = self.generate_synthetic_shapes(n_samples, **kwargs)
        
        elif dataset_type == "medical":
            images = self.generate_medical_images(n_samples, **kwargs)
        
        elif dataset_type == "satellite":
            images = self.generate_satellite_images(n_samples, **kwargs)
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Save dataset
        np.save(dataset_path, images)
        
        # Update metadata
        self.datasets[dataset_name] = {
            "type": dataset_type,
            "path": str(dataset_path),
            "n_samples": n_samples,
            "shape": images.shape,
            "created_at": str(Path().cwd()),
            "parameters": kwargs
        }
        
        self.save_metadata()
        
        return str(dataset_path)
    
    def load_dataset(self, dataset_name: str) -> np.ndarray:
        """Load a dataset by name"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset_path = self.datasets[dataset_name]["path"]
        return np.load(dataset_path)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets"""
        return [
            {
                "name": name,
                "type": info["type"],
                "n_samples": info["n_samples"],
                "shape": info["shape"]
            }
            for name, info in self.datasets.items()
        ]
    
    def visualize_dataset(self, dataset_name: str, n_samples: int = 16) -> None:
        """Visualize samples from a dataset"""
        images = self.load_dataset(dataset_name)
        
        # Select random samples
        indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
        sample_images = images[indices]
        
        # Create visualization
        n_cols = 4
        n_rows = (len(sample_images) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, img in enumerate(sample_images):
            row = i // n_cols
            col = i % n_cols
            
            if len(img.shape) == 3:
                axes[row, col].imshow(img)
            else:
                axes[row, col].imshow(img, cmap='gray')
            
            axes[row, col].set_title(f"Sample {i+1}")
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(len(sample_images), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f"Dataset: {dataset_name}")
        plt.tight_layout()
        plt.show()
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get detailed information about a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        return self.datasets[dataset_name]


def create_sample_datasets():
    """Create sample datasets for demonstration"""
    db = MockDatabase()
    
    # Create various sample datasets
    datasets_to_create = [
        ("geometric_shapes", "synthetic", 1000, {"image_size": (64, 64)}),
        ("text_samples", "synthetic", 500, {"image_size": (64, 64)}),
        ("medical_xray", "medical", 300, {"image_size": (128, 128)}),
        ("satellite_imagery", "satellite", 400, {"image_size": (64, 64)}),
    ]
    
    print("Creating sample datasets...")
    for name, dtype, n_samples, kwargs in datasets_to_create:
        try:
            path = db.create_dataset(name, dtype, n_samples, **kwargs)
            print(f"✓ Created {name}: {n_samples} samples at {path}")
        except Exception as e:
            print(f"✗ Failed to create {name}: {e}")
    
    print("\nAvailable datasets:")
    for dataset_info in db.list_datasets():
        print(f"  - {dataset_info['name']}: {dataset_info['n_samples']} samples, shape {dataset_info['shape']}")


if __name__ == "__main__":
    create_sample_datasets()

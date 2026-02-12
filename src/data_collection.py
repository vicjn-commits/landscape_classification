"""
Landscape Image Classification Pipeline
Data Collection and Preprocessing Module
"""

import os
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import json
import time

class LandscapeDataCollector:
    """
    Collects landscape images from various sources.
    In a real scenario, you'd use APIs like Unsplash, Pexels, or Flickr.
    For this demo, we'll create synthetic data and show the structure.
    """
    
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.categories = [
            'mountains', 'glaciers', 'prairies', 'desert', 'forest',
            'waterfalls', 'canyons', 'beaches', 'lakes', 'rivers',
            'valleys', 'plateaus', 'cliffs', 'dunes', 'tundra',
            'hills', 'caves', 'volcanoes', 'islands', 'fjords'
        ]
        self.images_per_category = 250  # Total: 5000 images
        
    def create_directory_structure(self):
        """Create organized directory structure for dataset"""
        paths = {
            'raw': os.path.join(self.base_dir, 'raw'),
            'processed': os.path.join(self.base_dir, 'processed'),
            'train': os.path.join(self.base_dir, 'processed', 'train'),
            'val': os.path.join(self.base_dir, 'processed', 'val'),
            'test': os.path.join(self.base_dir, 'processed', 'test')
        }
        
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
            
        # Create category subdirectories
        for split in ['train', 'val', 'test']:
            for category in self.categories:
                os.makedirs(
                    os.path.join(self.base_dir, 'processed', split, category),
                    exist_ok=True
                )
        
        return paths
    
    def generate_sample_images(self, save_path):
        """
        Generate sample images for demonstration.
        In production, replace this with actual API calls.
        """
        print("Generating sample dataset structure...")
        
        metadata = []
        
        for idx, category in enumerate(tqdm(self.categories, desc="Categories")):
            category_path = os.path.join(save_path, category)
            os.makedirs(category_path, exist_ok=True)
            
            for img_num in range(min(10, self.images_per_category)):  # Generate 10 samples per category
                # Create a simple gradient image as placeholder
                # In real scenario, download from API
                img = self._create_placeholder_image(category, idx)
                
                filename = f"{category}_{img_num:04d}.jpg"
                filepath = os.path.join(category_path, filename)
                img.save(filepath, 'JPEG')
                
                metadata.append({
                    'filename': filename,
                    'category': category,
                    'category_id': idx,
                    'width': img.width,
                    'height': img.height,
                    'split': None  # Will be assigned during train/val/test split
                })
        
        # Save metadata
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(self.base_dir, 'metadata.csv'), index=False)
        
        return df
    
    def _create_placeholder_image(self, category, idx):
        """Create a colored placeholder image"""
        # Create an image with category-specific colors
        np.random.seed(hash(category) % 2**32)
        
        # Generate random gradient
        width, height = 224, 224
        color_base = np.array([idx * 12 % 255, (idx * 37) % 255, (idx * 73) % 255])
        
        # Create gradient image
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(3):
            img_array[:, :, i] = (color_base[i] * (xx + yy) / 2).astype(np.uint8)
        
        # Add some noise for texture
        noise = np.random.randint(-30, 30, (height, width, 3))
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def download_from_unsplash(self, access_key):
        """
        Template for downloading from Unsplash API
        Requires Unsplash API key
        """
        # This is a template - you need to add your Unsplash API key
        base_url = "https://api.unsplash.com/search/photos"
        
        print("NOTE: To use real images, you need to:")
        print("1. Get an Unsplash API key from https://unsplash.com/developers")
        print("2. Use this template to download actual images")
        print("\nTemplate code structure:")
        print("""
        headers = {'Authorization': f'Client-ID {access_key}'}
        params = {
            'query': category,
            'per_page': 30,
            'orientation': 'landscape'
        }
        response = requests.get(base_url, headers=headers, params=params)
        images = response.json()['results']
        """)


if __name__ == "__main__":
    collector = LandscapeDataCollector()
    paths = collector.create_directory_structure()
    
    # Generate sample dataset
    print("\n" + "="*60)
    print("LANDSCAPE CLASSIFICATION DATASET PREPARATION")
    print("="*60 + "\n")
    
    metadata = collector.generate_sample_images(os.path.join('data', 'raw'))
    
    print(f"\n✓ Dataset structure created")
    print(f"✓ Total categories: {len(collector.categories)}")
    print(f"✓ Sample images generated: {len(metadata)}")
    print(f"\nCategories: {', '.join(collector.categories)}")
    
    # Display instructions for real data collection
    print("\n" + "="*60)
    print("TO USE REAL IMAGES:")
    print("="*60)
    print("1. Sign up for Unsplash/Pexels/Flickr API")
    print("2. Use the download_from_unsplash() template")
    print("3. Or use existing datasets like:")
    print("   - Intel Image Classification (Kaggle)")
    print("   - Landscape Pictures (Kaggle)")
    print("   - Places365 Dataset")

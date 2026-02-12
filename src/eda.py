"""
Exploratory Data Analysis Module
Comprehensive analysis of landscape image dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

class LandscapeEDA:
    """
    Performs comprehensive EDA on landscape image dataset
    """
    
    def __init__(self, data_dir='data/raw', metadata_path='data/metadata.csv'):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else None
        self.results_dir = 'results/eda'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def analyze_dataset_structure(self):
        """Analyze basic dataset structure"""
        print("\n" + "="*60)
        print("DATASET STRUCTURE ANALYSIS")
        print("="*60 + "\n")
        
        if self.metadata is not None:
            print(f"Total images: {len(self.metadata)}")
            print(f"\nClass distribution:")
            print(self.metadata['category'].value_counts().sort_index())
            
            # Check for class imbalance
            class_counts = self.metadata['category'].value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 1.5:
                print("⚠ Warning: Significant class imbalance detected")
            else:
                print("✓ Dataset is relatively balanced")
                
        return self.metadata
    
    def analyze_image_properties(self):
        """Analyze image dimensions, aspect ratios, file sizes"""
        print("\n" + "="*60)
        print("IMAGE PROPERTIES ANALYSIS")
        print("="*60 + "\n")
        
        dimensions = []
        aspect_ratios = []
        file_sizes = []
        
        for category in os.listdir(self.data_dir):
            category_path = os.path.join(self.data_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            for img_file in os.listdir(category_path):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(category_path, img_file)
                
                # Get file size
                file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
                
                # Get dimensions
                img = Image.open(img_path)
                width, height = img.size
                dimensions.append((width, height))
                aspect_ratios.append(width / height)
        
        # Statistics
        dimensions_array = np.array(dimensions)
        print(f"Image dimensions:")
        print(f"  Width  - Mean: {dimensions_array[:, 0].mean():.1f}, "
              f"Std: {dimensions_array[:, 0].std():.1f}")
        print(f"  Height - Mean: {dimensions_array[:, 1].mean():.1f}, "
              f"Std: {dimensions_array[:, 1].std():.1f}")
        
        print(f"\nAspect ratios:")
        print(f"  Mean: {np.mean(aspect_ratios):.2f}")
        print(f"  Std: {np.std(aspect_ratios):.2f}")
        
        print(f"\nFile sizes (KB):")
        print(f"  Mean: {np.mean(file_sizes):.1f}")
        print(f"  Min: {np.min(file_sizes):.1f}, Max: {np.max(file_sizes):.1f}")
        
        return {
            'dimensions': dimensions_array,
            'aspect_ratios': aspect_ratios,
            'file_sizes': file_sizes
        }
    
    def analyze_color_distribution(self, sample_size=100):
        """Analyze color distributions across categories"""
        print("\n" + "="*60)
        print("COLOR DISTRIBUTION ANALYSIS")
        print("="*60 + "\n")
        
        color_stats = {}
        
        for category in os.listdir(self.data_dir):
            category_path = os.path.join(self.data_dir, category)
            if not os.path.isdir(category_path):
                continue
            
            rgb_means = []
            hsv_means = []
            
            files = [f for f in os.listdir(category_path) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in files[:min(sample_size, len(files))]:
                img_path = os.path.join(category_path, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # RGB analysis
                rgb_mean = img.mean(axis=(0, 1))
                rgb_means.append(rgb_mean)
                
                # HSV analysis
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv_mean = hsv.mean(axis=(0, 1))
                hsv_means.append(hsv_mean)
            
            if rgb_means:
                color_stats[category] = {
                    'rgb_mean': np.mean(rgb_means, axis=0),
                    'rgb_std': np.std(rgb_means, axis=0),
                    'hsv_mean': np.mean(hsv_means, axis=0),
                    'hsv_std': np.std(hsv_means, axis=0)
                }
        
        # Print summary
        for category, stats in list(color_stats.items())[:5]:
            print(f"\n{category}:")
            print(f"  RGB mean: {stats['rgb_mean']}")
            print(f"  HSV mean: {stats['hsv_mean']}")
        
        return color_stats
    
    def create_visualizations(self, properties):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Class distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if self.metadata is not None:
            # Class counts
            self.metadata['category'].value_counts().plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Category')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Image dimensions
        axes[0, 1].scatter(properties['dimensions'][:, 0], 
                          properties['dimensions'][:, 1], alpha=0.5)
        axes[0, 1].set_title('Image Dimensions Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Width (pixels)')
        axes[0, 1].set_ylabel('Height (pixels)')
        
        # Aspect ratios
        axes[1, 0].hist(properties['aspect_ratios'], bins=30, edgecolor='black')
        axes[1, 0].set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Aspect Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(x=np.mean(properties['aspect_ratios']), 
                          color='r', linestyle='--', label='Mean')
        axes[1, 0].legend()
        
        # File sizes
        axes[1, 1].hist(properties['file_sizes'], bins=30, edgecolor='black')
        axes[1, 1].set_title('File Size Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('File Size (KB)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/eda_overview.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.results_dir}/eda_overview.png")
        plt.close()
        
    def generate_report(self):
        """Generate comprehensive EDA report"""
        print("\n" + "="*60)
        print("GENERATING EDA REPORT")
        print("="*60 + "\n")
        
        # Run all analyses
        self.analyze_dataset_structure()
        properties = self.analyze_image_properties()
        color_stats = self.analyze_color_distribution(sample_size=20)
        self.create_visualizations(properties)
        
        # Save summary report
        report = {
            'total_images': len(self.metadata) if self.metadata is not None else 0,
            'num_classes': len(self.metadata['category'].unique()) if self.metadata is not None else 0,
            'avg_width': float(properties['dimensions'][:, 0].mean()),
            'avg_height': float(properties['dimensions'][:, 1].mean()),
            'avg_aspect_ratio': float(np.mean(properties['aspect_ratios'])),
            'avg_file_size_kb': float(np.mean(properties['file_sizes']))
        }
        
        with open(f'{self.results_dir}/eda_summary.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ EDA complete. Results saved to {self.results_dir}/")
        return report


if __name__ == "__main__":
    eda = LandscapeEDA()
    report = eda.generate_report()

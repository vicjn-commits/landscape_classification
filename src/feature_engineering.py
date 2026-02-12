"""
Feature Engineering Module
Extracts various features from landscape images for ML models
"""

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

class LandscapeFeatureExtractor:
    """
    Extracts multiple types of features from landscape images:
    1. Color features (histograms, moments)
    2. Texture features (GLCM, LBP)
    3. Edge features (Canny, Sobel)
    4. HOG features
    5. Deep features (from pre-trained models)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_color_features(self, img):
        """Extract color-based features"""
        features = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Color moments (mean, std, skewness) for each channel
        for color_space, name in [(img, 'BGR'), (hsv, 'HSV'), (lab, 'LAB')]:
            for i in range(3):
                channel = color_space[:, :, i]
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    self._skewness(channel)
                ])
        
        # Color histograms
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        return np.array(features)
    
    def extract_texture_features(self, img):
        """Extract texture features using various methods"""
        features = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern (simplified version)
        lbp = self._compute_lbp(gray)
        lbp_hist, _ = np.histogram(lbp, bins=32, range=(0, 256))
        features.extend(lbp_hist)
        
        # Gabor filters
        gabor_features = self._compute_gabor_features(gray)
        features.extend(gabor_features)
        
        # Haralick texture features (simplified GLCM)
        glcm_features = self._compute_glcm_features(gray)
        features.extend(glcm_features)
        
        return np.array(features)
    
    def extract_edge_features(self, img):
        """Extract edge-based features"""
        features = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        features.extend([
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0) / edges.size  # Edge density
        ])
        
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        features.extend([
            np.mean(np.abs(sobelx)),
            np.mean(np.abs(sobely)),
            np.std(sobelx),
            np.std(sobely)
        ])
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.percentile(magnitude, 90)
        ])
        
        return np.array(features)
    
    def extract_hog_features(self, img):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray_resized = cv2.resize(gray, (128, 128))
        
        # Compute HOG
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(gray_resized)
        
        return hog_features.flatten()
    
    def extract_shape_features(self, img):
        """Extract shape-based features"""
        features = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Compute moments
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                features.extend([
                    moments['m00'],  # Area
                    moments['m10'] / moments['m00'],  # Centroid x
                    moments['m01'] / moments['m00'],  # Centroid y
                ])
            else:
                features.extend([0, 0, 0])
            
            # Perimeter and compactness
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                features.append(compactness)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def extract_all_features(self, img_path):
        """Extract all features from an image"""
        img = cv2.imread(img_path)
        
        if img is None:
            return None
        
        # Resize for consistency
        img = cv2.resize(img, (224, 224))
        
        # Extract different feature types
        color_feat = self.extract_color_features(img)
        texture_feat = self.extract_texture_features(img)
        edge_feat = self.extract_edge_features(img)
        hog_feat = self.extract_hog_features(img)
        shape_feat = self.extract_shape_features(img)
        
        # Concatenate all features
        all_features = np.concatenate([
            color_feat,
            texture_feat,
            edge_feat,
            hog_feat[:100],  # Reduce HOG dimension
            shape_feat
        ])
        
        return all_features
    
    def process_dataset(self, data_dir, metadata_path, output_path='data/features.npz'):
        """Process entire dataset and extract features"""
        print("\n" + "="*60)
        print("FEATURE EXTRACTION")
        print("="*60 + "\n")
        
        metadata = pd.read_csv(metadata_path)
        
        all_features = []
        all_labels = []
        all_filenames = []
        
        for category in tqdm(os.listdir(data_dir), desc="Processing categories"):
            category_path = os.path.join(data_dir, category)
            if not os.path.isdir(category_path):
                continue
            
            for img_file in os.listdir(category_path):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(category_path, img_file)
                
                # Extract features
                features = self.extract_all_features(img_path)
                
                if features is not None:
                    all_features.append(features)
                    all_labels.append(category)
                    all_filenames.append(img_file)
        
        # Convert to arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Save features and labels
        np.savez(output_path, 
                 features=X_scaled, 
                 labels=y,
                 filenames=all_filenames)
        
        # Save scaler
        with open('data/feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n✓ Extracted features from {len(all_features)} images")
        print(f"✓ Feature dimension: {X.shape[1]}")
        print(f"✓ Saved to: {output_path}")
        
        return X_scaled, y, all_filenames
    
    # Helper methods
    def _skewness(self, arr):
        """Calculate skewness of array"""
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0
        return np.mean(((arr - mean) / std) ** 3)
    
    def _compute_lbp(self, gray):
        """Compute Local Binary Pattern"""
        lbp = np.zeros_like(gray)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp[i, j] = code
        return lbp
    
    def _compute_gabor_features(self, gray):
        """Compute Gabor filter responses"""
        features = []
        ksize = 21
        sigma = 5
        
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            features.extend([np.mean(filtered), np.std(filtered)])
        
        return features
    
    def _compute_glcm_features(self, gray):
        """Compute simplified GLCM features"""
        # Reduce to 32 gray levels for efficiency
        gray_reduced = (gray / 8).astype(np.uint8)
        
        # Compute co-occurrence matrix
        glcm = np.zeros((32, 32))
        for i in range(gray_reduced.shape[0]-1):
            for j in range(gray_reduced.shape[1]-1):
                glcm[gray_reduced[i, j], gray_reduced[i, j+1]] += 1
        
        # Normalize
        glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
        
        # Compute features
        contrast = np.sum([((i-j)**2) * glcm[i, j] for i in range(32) for j in range(32)])
        homogeneity = np.sum([glcm[i, j] / (1 + abs(i-j)) for i in range(32) for j in range(32)])
        energy = np.sum(glcm ** 2)
        
        return [contrast, homogeneity, energy]


if __name__ == "__main__":
    extractor = LandscapeFeatureExtractor()
    X, y, filenames = extractor.process_dataset('data/raw', 'data/metadata.csv')

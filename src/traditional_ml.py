"""
Traditional ML Models Module
Implements Logistic Regression and other baseline models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json


class TraditionalMLPipeline:
    """
    Traditional ML pipeline using engineered features
    """
    
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.results = {}
        
    def load_features(self, features_path='data/features.npz'):
        """Load extracted features"""
        data = np.load(features_path, allow_pickle=True)
        X = data['features']
        y = data['labels']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train Logistic Regression with hyperparameter tuning"""
        print("\n" + "="*60)
        print("LOGISTIC REGRESSION")
        print("="*60 + "\n")
        
        # Define hyperparameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [500, 1000]
        }
        
        # Grid search with cross-validation
        lr = LogisticRegression(random_state=42, multi_class='multinomial')
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='f1_macro', 
            n_jobs=-1, verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on validation set
        best_lr = grid_search.best_estimator_
        val_accuracy = best_lr.score(X_val, y_val)
        val_predictions = best_lr.predict(X_val)
        val_f1 = f1_score(y_val, val_predictions, average='macro')
        
        print(f"\nValidation accuracy: {val_accuracy:.4f}")
        print(f"Validation F1-score: {val_f1:.4f}")
        
        self.models['logistic_regression'] = best_lr
        self.results['logistic_regression'] = {
            'best_params': grid_search.best_params_,
            'cv_score': float(grid_search.best_score_),
            'val_accuracy': float(val_accuracy),
            'val_f1': float(val_f1)
        }
        
        return best_lr
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest classifier"""
        print("\n" + "="*60)
        print("RANDOM FOREST")
        print("="*60 + "\n")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='f1_macro', 
            n_jobs=-1, verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        best_rf = grid_search.best_estimator_
        val_accuracy = best_rf.score(X_val, y_val)
        val_predictions = best_rf.predict(X_val)
        val_f1 = f1_score(y_val, val_predictions, average='macro')
        
        print(f"\nValidation accuracy: {val_accuracy:.4f}")
        print(f"Validation F1-score: {val_f1:.4f}")
        
        self.models['random_forest'] = best_rf
        self.results['random_forest'] = {
            'best_params': grid_search.best_params_,
            'cv_score': float(grid_search.best_score_),
            'val_accuracy': float(val_accuracy),
            'val_f1': float(val_f1)
        }
        
        return best_rf
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print('='*60 + "\n")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save results
        self.results[model_name]['test_metrics'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, model_name, save_path='results/'):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60 + "\n")
        
        comparison_df = pd.DataFrame({
            model_name: {
                'Val Accuracy': metrics['val_accuracy'],
                'Val F1': metrics['val_f1'],
                'Test Accuracy': metrics.get('test_metrics', {}).get('accuracy', 0),
                'Test F1': metrics.get('test_metrics', {}).get('f1_score', 0)
            }
            for model_name, metrics in self.results.items()
        }).T
        
        print(comparison_df.to_string())
        
        # Find best model
        best_model_name = comparison_df['Test F1'].idxmax()
        self.best_model = self.models[best_model_name]
        
        print(f"\n✓ Best model: {best_model_name}")
        print(f"  Test F1-score: {comparison_df.loc[best_model_name, 'Test F1']:.4f}")
        
        return comparison_df
    
    def save_models(self, path='models/'):
        """Save trained models and results"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            with open(f'{path}/{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save label encoder
        with open(f'{path}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save results
        with open(f'{path}/results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Models saved to {path}")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print(" "*15 + "TRADITIONAL ML PIPELINE")
    print("="*70 + "\n")
    
    pipeline = TraditionalMLPipeline()
    
    # Load features
    print("Loading features...")
    X, y_encoded, y_labels = pipeline.load_features()
    print(f"✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"✓ Number of classes: {len(np.unique(y_encoded))}\n")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(X, y_encoded)
    
    # Train models
    lr_model = pipeline.train_logistic_regression(X_train, y_train, X_val, y_val)
    rf_model = pipeline.train_random_forest(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    lr_results = pipeline.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
    rf_results = pipeline.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    
    # Plot confusion matrices
    os.makedirs('results', exist_ok=True)
    pipeline.plot_confusion_matrix(lr_results['confusion_matrix'], 'Logistic Regression')
    pipeline.plot_confusion_matrix(rf_results['confusion_matrix'], 'Random Forest')
    
    # Compare models
    comparison = pipeline.compare_models()
    
    # Save everything
    pipeline.save_models()
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

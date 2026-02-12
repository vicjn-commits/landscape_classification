#!/usr/bin/env python3
"""
Master Pipeline Runner
Execute the complete landscape classification pipeline
"""

import os
import sys
import argparse
import time
from datetime import datetime

def setup_environment():
    """Set up environment and paths"""
    print("\n" + "="*80)
    print(" " * 20 + "LANDSCAPE CLASSIFICATION PIPELINE")
    print("="*80 + "\n")
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed/train',
        'data/processed/val', 
        'data/processed/test',
        'models/checkpoints',
        'results/eda',
        'results/confusion_matrices',
        'results/training_histories',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Environment setup complete\n")


def run_data_collection():
    """Step 1: Data Collection"""
    print("\n" + "-"*80)
    print("STEP 1: DATA COLLECTION")
    print("-"*80 + "\n")
    
    from src.data_collection import LandscapeDataCollector
    
    collector = LandscapeDataCollector(base_dir='data')
    collector.create_directory_structure()
    metadata = collector.generate_sample_images(os.path.join('data', 'raw'))
    
    print(f"\n✓ Step 1 Complete: {len(metadata)} images prepared")
    return metadata


def run_eda():
    """Step 2: Exploratory Data Analysis"""
    print("\n" + "-"*80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("-"*80 + "\n")
    
    from src.eda import LandscapeEDA
    
    eda = LandscapeEDA(data_dir='data/raw', metadata_path='data/metadata.csv')
    report = eda.generate_report()
    
    print("\n✓ Step 2 Complete: EDA report generated")
    return report


def run_feature_engineering():
    """Step 3: Feature Engineering"""
    print("\n" + "-"*80)
    print("STEP 3: FEATURE ENGINEERING")
    print("-"*80 + "\n")
    
    from src.feature_engineering import LandscapeFeatureExtractor
    
    extractor = LandscapeFeatureExtractor()
    X, y, filenames = extractor.process_dataset(
        data_dir='data/raw',
        metadata_path='data/metadata.csv',
        output_path='data/features.npz'
    )
    
    print(f"\n✓ Step 3 Complete: Extracted {X.shape[1]} features from {X.shape[0]} images")
    return X, y, filenames


def run_traditional_ml():
    """Step 4: Traditional ML Models"""
    print("\n" + "-"*80)
    print("STEP 4: TRADITIONAL MACHINE LEARNING")
    print("-"*80 + "\n")
    
    from src.traditional_ml import TraditionalMLPipeline
    
    pipeline = TraditionalMLPipeline()
    
    # Load features
    X, y_encoded, y_labels = pipeline.load_features('data/features.npz')
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(X, y_encoded)
    
    # Train models
    print("\nTraining Logistic Regression...")
    lr_model = pipeline.train_logistic_regression(X_train, y_train, X_val, y_val)
    
    print("\nTraining Random Forest...")
    rf_model = pipeline.train_random_forest(X_train, y_train, X_val, y_val)
    
    # Evaluate
    lr_results = pipeline.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
    rf_results = pipeline.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    
    # Plot results
    os.makedirs('results/confusion_matrices', exist_ok=True)
    pipeline.plot_confusion_matrix(
        lr_results['confusion_matrix'], 
        'Logistic Regression', 
        'results/confusion_matrices/'
    )
    pipeline.plot_confusion_matrix(
        rf_results['confusion_matrix'], 
        'Random Forest', 
        'results/confusion_matrices/'
    )
    
    # Compare
    comparison = pipeline.compare_models()
    
    # Save models
    pipeline.save_models('models/')
    
    print("\n✓ Step 4 Complete: Traditional ML models trained and evaluated")
    return comparison


def run_deep_learning():
    """Step 5: Deep Learning Models"""
    print("\n" + "-"*80)
    print("STEP 5: DEEP LEARNING (CNN)")
    print("-"*80 + "\n")
    
    from src.cnn_models import CNNPipeline
    
    print("NOTE: Deep learning training requires:")
    print("  1. Organized image data in data/processed/train, val, test")
    print("  2. Significant computational resources (GPU recommended)")
    print("  3. Extended training time (1-2 hours per model)")
    print("\nFor demonstration, we'll build the model architectures only.")
    
    pipeline = CNNPipeline(img_size=(224, 224), num_classes=20)
    
    # Build models
    print("\nBuilding Custom CNN...")
    custom_cnn = pipeline.build_custom_cnn()
    
    print("\nBuilding Transfer Learning Models...")
    resnet_model = pipeline.build_transfer_learning_model('resnet50')
    
    # Save architectures
    with open('models/custom_cnn_architecture.json', 'w') as f:
        f.write(custom_cnn.to_json())
    
    print("\n✓ Step 5 Complete: CNN architectures built")
    print("\nTo train these models, run:")
    print("  python src/cnn_models.py")
    
    return pipeline


def generate_final_report(metadata, eda_report, ml_comparison):
    """Generate final summary report"""
    print("\n" + "="*80)
    print(" " * 30 + "FINAL REPORT")
    print("="*80 + "\n")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'total_images': len(metadata),
            'num_classes': len(metadata['category'].unique()),
        },
        'eda': eda_report,
        'models': {
            'traditional_ml': ml_comparison.to_dict() if hasattr(ml_comparison, 'to_dict') else str(ml_comparison),
        }
    }
    
    # Save report
    import json
    with open('results/final_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("Pipeline Summary:")
    print(f"  • Total Images: {len(metadata)}")
    print(f"  • Classes: {len(metadata['category'].unique())}")
    print(f"  • Best Traditional ML Model: Logistic Regression or Random Forest")
    print(f"  • Features Extracted: ~200 per image")
    print("\nResults saved to:")
    print("  • results/eda/")
    print("  • results/confusion_matrices/")
    print("  • models/")
    print("  • results/final_report.json")
    
    print("\n✓ Final report generated: results/final_report.json")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Landscape Classification Pipeline')
    parser.add_argument('--steps', nargs='+', 
                       choices=['all', 'data', 'eda', 'features', 'ml', 'dl'],
                       default=['all'],
                       help='Steps to run (default: all)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Setup
    setup_environment()
    
    # Determine steps to run
    steps = args.steps
    run_all = 'all' in steps
    
    # Execute pipeline
    metadata = None
    eda_report = None
    ml_comparison = None
    
    if run_all or 'data' in steps:
        metadata = run_data_collection()
    
    if run_all or 'eda' in steps:
        eda_report = run_eda()
    
    if run_all or 'features' in steps:
        X, y, filenames = run_feature_engineering()
    
    if run_all or 'ml' in steps:
        ml_comparison = run_traditional_ml()
    
    if run_all or 'dl' in steps:
        run_deep_learning()
    
    # Generate final report
    if run_all:
        import pandas as pd
        if metadata is None:
            metadata = pd.read_csv('data/metadata.csv')
        if eda_report is None:
            import json
            with open('results/eda/eda_summary.json', 'r') as f:
                eda_report = json.load(f)
        if ml_comparison is None:
            ml_comparison = "See models/results.json"
        
        generate_final_report(metadata, eda_report, ml_comparison)
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(" " * 25 + "PIPELINE EXECUTION COMPLETE!")
    print("="*80)
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")
    print("\nNext Steps:")
    print("  1. Review results in results/ directory")
    print("  2. Examine notebooks/complete_pipeline.ipynb for detailed analysis")
    print("  3. Train deep learning models for production use")
    print("  4. Deploy best model using deployment scripts")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

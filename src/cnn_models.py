"""
Deep Learning CNN Module
Implements CNN models with transfer learning and custom architecture
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import (
    ResNet50, VGG16, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json


class CNNPipeline:
    """
    CNN pipeline with multiple architectures and transfer learning
    """
    
    def __init__(self, img_size=(224, 224), num_classes=20):
        self.img_size = img_size
        self.num_classes = num_classes
        self.models = {}
        self.history = {}
        self.label_encoder = LabelEncoder()
        
    def create_data_generators(self, data_dir, batch_size=32, validation_split=0.2):
        """Create data generators with augmentation"""
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Test data (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        val_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def build_custom_cnn(self):
        """Build custom CNN architecture from scratch"""
        print("\n" + "="*60)
        print("BUILDING CUSTOM CNN")
        print("="*60 + "\n")
        
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print(model.summary())
        self.models['custom_cnn'] = model
        return model
    
    def build_transfer_learning_model(self, base_model_name='resnet50', trainable_layers=0):
        """Build transfer learning model"""
        print(f"\n{'='*60}")
        print(f"BUILDING TRANSFER LEARNING MODEL: {base_model_name.upper()}")
        print('='*60 + "\n")
        
        # Load base model
        base_models = {
            'resnet50': ResNet50,
            'vgg16': VGG16,
            'mobilenetv2': MobileNetV2,
            'efficientnetb0': EfficientNetB0
        }
        
        base_model = base_models[base_model_name](
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # If trainable_layers > 0, unfreeze last n layers
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Build complete model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print(f"Total parameters: {model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        self.models[f'transfer_{base_model_name}'] = model
        return model
    
    def create_callbacks(self, model_name):
        """Create training callbacks"""
        os.makedirs('models/checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        callback_list = [
            # Model checkpoint
            callbacks.ModelCheckpoint(
                f'models/checkpoints/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir=f'logs/{model_name}',
                histogram_freq=1
            )
        ]
        
        return callback_list
    
    def train_model(self, model, train_gen, val_gen, model_name, epochs=50):
        """Train a model"""
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name.upper()}")
        print('='*60 + "\n")
        
        callback_list = self.create_callbacks(model_name)
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history[model_name] = history.history
        
        # Save training history
        with open(f'models/{model_name}_history.json', 'w') as f:
            json.dump(history.history, f, indent=2)
        
        print(f"\n✓ Training complete for {model_name}")
        return history
    
    def fine_tune_model(self, model, train_gen, val_gen, model_name, 
                       unfreeze_layers=30, epochs=20):
        """Fine-tune a transfer learning model"""
        print(f"\n{'='*60}")
        print(f"FINE-TUNING {model_name.upper()}")
        print('='*60 + "\n")
        
        # Unfreeze layers
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Freeze early layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        callback_list = self.create_callbacks(f'{model_name}_finetuned')
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history[f'{model_name}_finetuned'] = history.history
        
        return history
    
    def evaluate_model(self, model, test_gen, model_name):
        """Evaluate model on test set"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print('='*60 + "\n")
        
        # Predictions
        test_gen.reset()
        predictions = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes
        
        # Metrics
        test_loss, test_acc, test_top3 = model.evaluate(test_gen, verbose=0)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Top-3 Accuracy: {test_top3:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        class_names = list(test_gen.class_indices.keys())
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'loss': test_loss,
            'accuracy': test_acc,
            'top3_accuracy': test_top3,
            'predictions': predictions,
            'confusion_matrix': cm,
            'class_names': class_names
        }
    
    def plot_training_history(self, model_name, save_path='results/'):
        """Plot training history"""
        if model_name not in self.history:
            print(f"No history found for {model_name}")
            return
        
        history = self.history[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Train')
        axes[0, 0].plot(history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Train')
        axes[0, 1].plot(history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy
        if 'top_3_accuracy' in history:
            axes[1, 0].plot(history['top_3_accuracy'], label='Train')
            axes[1, 0].plot(history['val_top_3_accuracy'], label='Validation')
            axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, f'{model_name}\nTraining History', 
                       ha='center', va='center', fontsize=16, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/training_history_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved training history plot: {save_path}/training_history_{model_name}.png")
    
    def plot_confusion_matrix(self, cm, class_names, model_name, save_path='results/'):
        """Plot confusion matrix"""
        plt.figure(figsize=(14, 12))
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix_cnn_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved confusion matrix: {save_path}/confusion_matrix_cnn_{model_name}.png")


def main():
    """Main CNN training pipeline"""
    print("\n" + "="*70)
    print(" "*20 + "CNN TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Initialize pipeline
    pipeline = CNNPipeline(img_size=(224, 224), num_classes=20)
    
    # Note: This assumes you have organized data in data/processed/train, val, test
    print("NOTE: This script assumes data is organized in:")
    print("  - data/processed/train/")
    print("  - data/processed/val/")
    print("  - data/processed/test/")
    print("\nFor demonstration, we'll show the architecture setup.\n")
    
    # Build models
    custom_cnn = pipeline.build_custom_cnn()
    resnet_model = pipeline.build_transfer_learning_model('resnet50')
    
    print("\n✓ Models built successfully!")
    print("\nTo train the models, you would:")
    print("1. Organize your 5000 images into train/val/test folders")
    print("2. Create data generators")
    print("3. Run pipeline.train_model() for each model")
    print("4. Fine-tune transfer learning models")
    print("5. Evaluate and compare all models")
    
    # Save model architectures
    os.makedirs('models', exist_ok=True)
    with open('models/custom_cnn_architecture.json', 'w') as f:
        f.write(custom_cnn.to_json())
    
    print("\n✓ Model architectures saved!")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import mode
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def apply_spatial_smoothing(pred_map, window_size=5):
    """Apply majority filter for spatial smoothing"""
    def majority_filter(values):
        return mode(values, keepdims=True)[0][0]
    
    return generic_filter(pred_map, majority_filter, size=window_size)

def plot_prediction_comparison(rgb, ground_truth, pred_raw, pred_smooth, save_path):
    """Create side-by-side comparison plot"""
    plt.figure(figsize=(20, 5))
    
    # RGB
    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title('RGB Image')
    plt.axis('off')
    
    # Ground Truth
    plt.subplot(1, 4, 2)
    plt.imshow(ground_truth, cmap='tab20', vmin=1, vmax=18)
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Raw Prediction
    plt.subplot(1, 4, 3)
    plt.imshow(pred_raw, cmap='tab20', vmin=1, vmax=18)
    plt.title('Raw Prediction')
    plt.axis('off')
    
    # Smoothed Prediction
    plt.subplot(1, 4, 4)
    plt.imshow(pred_smooth, cmap='tab20', vmin=1, vmax=18)
    plt.title('Spatially Smoothed')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {save_path}")

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_class_distribution(class_counts, save_path):
    """Plot class distribution"""
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(classes, counts)
    plt.xlabel('Class')
    plt.ylabel('Pixel Count')
    plt.title('Class Distribution in Training Data')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(model, save_path):
    """Plot LightGBM feature importance"""
    importance = model.feature_importances_
    feature_names = [
        f'Band {i} Mean' for i in range(10)
    ] + [
        f'Band {i} Std' for i in range(10)
    ] + [
        f'Band {i} Max' for i in range(10)
    ] + [
        f'Band {i} Min' for i in range(10)
    ]
    
    # Sort by importance
    idx = np.argsort(importance)[-20:]  # Top 20
    plt.figure(figsize=(10, 8))
    plt.barh([feature_names[i] for i in idx], importance[idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Features by Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
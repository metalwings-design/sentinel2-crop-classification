import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def evaluate_model(model, X_val, y_val, save_dir="outputs/metrics"):
    """Generate classification report and confusion matrix"""
    y_pred = model.predict(X_val)
    
    # Metrics with zero_division=0 to suppress warnings
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)
    
    # Save metrics
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report_str)
    
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)
    
    return y_pred, accuracy, report, cm
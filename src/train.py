import lightgbm as lgb
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def train_model(X_train, y_train, config):
    """Train LightGBM model"""
    model_params = config['model']['params']
    model = lgb.LGBMClassifier(**model_params)
    model.fit(X_train, y_train)
    return model

def save_model(model, path="outputs/models/lightgbm_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path="outputs/models/lightgbm_model.pkl"):
    return joblib.load(path)

def evaluate_model(model, X_val, y_val, save_dir="outputs/metrics"):
    """Generate classification report and confusion matrix"""
    y_pred = model.predict(X_val)
    
    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)
    
    # Save metrics
    os.makedirs(save_dir, exist_ok=True)
    
    # Save classification report as text
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_val, y_pred, zero_division=0))
    
    # Save confusion matrix
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)
    
    return y_pred, accuracy, report, cm
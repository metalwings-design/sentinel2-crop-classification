import os
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_data_splits(s2_dir, random_state=42):
    """Create train/val/test splits from patch IDs"""
    s2_files = sorted([f for f in os.listdir(s2_dir) if f.endswith('.npy')])
    patch_ids = [f.split('_')[1].split('.')[0] for f in s2_files]
    
    train_ids, temp_ids = train_test_split(patch_ids, test_size=0.3, random_state=random_state)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=random_state)
    
    return train_ids, val_ids, test_ids

def load_patch(patch_id, s2_dir, ann_dir):
    """Load S2 and annotation for a patch"""
    s2 = np.load(os.path.join(s2_dir, f"S2_{patch_id}.npy"))
    target = np.load(os.path.join(ann_dir, f"TARGET_{patch_id}.npy"))
    return s2, target

def save_splits(train_ids, val_ids, test_ids, splits_dir="splits"):
    """Save patch IDs to text files for reproducibility"""
    os.makedirs(splits_dir, exist_ok=True)
    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_ids))
    with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_ids))
    with open(os.path.join(splits_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_ids))
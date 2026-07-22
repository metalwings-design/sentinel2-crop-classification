import numpy as np
import os

def prepare_data(s2_data, target):
    """
    Extract temporal features from Sentinel-2 time series.
    Returns: 40 features per pixel (10 bands × 4 statistics)
    """
    # Normalize each band to [0,1]
    s2_min = s2_data.min(axis=(0,2,3), keepdims=True)
    s2_max = s2_data.max(axis=(0,2,3), keepdims=True)
    s2_norm = (s2_data - s2_min) / (s2_max - s2_min + 1e-8)
    
    # Mask out background (0) and void (19)
    mask = (target[0] != 0) & (target[0] != 19)
    y = target[0][mask]
    
    # Extract 4 temporal statistics per band
    feat_mean = s2_norm.mean(axis=0)      # (10, 128, 128)
    feat_std = s2_norm.std(axis=0)        # (10, 128, 128)
    feat_max = s2_norm.max(axis=0)        # (10, 128, 128)
    feat_min = s2_norm.min(axis=0)        # (10, 128, 128)
    
    # Concatenate to 40 features
    X = np.concatenate([feat_mean, feat_std, feat_max, feat_min], axis=0)
    X = X[:, mask].T  # (n_pixels, 40)
    
    return X, y, mask

def build_dataset(patch_ids, s2_dir, ann_dir, max_patches=None, pixels_per_patch=None):
    """Build training dataset from multiple patches"""
    X_list, y_list = [], []
    
    if max_patches:
        patch_ids = patch_ids[:max_patches]
    
    for patch_id in patch_ids:
        s2 = np.load(os.path.join(s2_dir, f"S2_{patch_id}.npy"))
        target = np.load(os.path.join(ann_dir, f"TARGET_{patch_id}.npy"))
        X, y, _ = prepare_data(s2, target)
        
        # Sample random pixels per patch
        if pixels_per_patch and len(y) > pixels_per_patch:
            idx = np.random.choice(len(y), pixels_per_patch, replace=False)
            X, y = X[idx], y[idx]
        
        X_list.append(X)
        y_list.append(y)
    
    return np.vstack(X_list), np.hstack(y_list)

def load_metadata(metadata_path):
    """Load temporal metadata"""
    import geopandas as gpd
    return gpd.read_file(metadata_path)
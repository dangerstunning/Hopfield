import numpy as np
import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    labels = df['label'].to_numpy()
    images = df.drop(columns=['label']).to_numpy()
    return images, labels

def preprocess_images(images, threshold=127):
    return np.where(images > threshold, 1, -1).astype(np.int8)

def create_variants(patterns, labels):
    return {
        'A': (patterns[0:70], labels[0:70]),
        'B': (patterns[70:140], labels[70:140]),
        'C': (patterns[140:210], labels[140:210]),
    }

def add_noise(pattern, noise_level, seed=None):
    rng = np.random.default_rng(seed)
    noisy = pattern.copy()
    n_flips = int(len(pattern) * noise_level)
    if n_flips:
        idx = rng.choice(len(pattern), n_flips, replace=False)
        noisy[idx] *= -1
    return noisy

import os
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import add_noise

def show_recall(model, patterns, labels, variant, sample_index=0, noise_level=0.30):
    original = patterns[sample_index]
    noisy = add_noise(original, noise_level, seed=42 + sample_index)
    recalled, iterations, _ = model.recall(noisy)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, image, title in [
        (axes[0], original, f'Original\nDigit: {labels[sample_index]}'),
        (axes[1], noisy, f'Noisy\n{int(noise_level*100)}%'),
        (axes[2], recalled, f'Recalled\n{iterations} iterations')]:
        ax.imshow(image.reshape(28, 28), cmap='gray')
        ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    os.makedirs(f'results/variant_{variant}', exist_ok=True)
    path = f'results/variant_{variant}/recall_{sample_index}_noise_{int(noise_level*100)}.png'
    plt.savefig(path, dpi=200); plt.close(fig)
    return path

def plot_metric(df, column, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    for variant in ['A', 'B', 'C']:
        data = df[df['Variant'] == variant]
        plt.plot(data['Noise'] * 100, data[column] * 100 if 'Accuracy' in column else data[column], marker='o', label=f'Variant {variant}')
    plt.xlabel('Noise Level (%)'); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout()
    path = f'results/{filename}'
    plt.savefig(path, dpi=200); plt.close(); return path

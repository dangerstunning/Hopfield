import os
import pandas as pd
from preprocessing import load_dataset, preprocess_images, create_variants
from experiment import run_experiment
from visualization import show_recall, plot_metric

DATA_PATH = 'data/train.csv'

def main():
    os.makedirs('results', exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print('ERROR: data/train.csv was not found.')
        print('Download train.csv from Kaggle Digit Recognizer and place it in the data folder.')
        return
    images, labels = load_dataset(DATA_PATH)
    if len(images) < 210 or images.shape[1] != 784:
        raise ValueError(f'Expected at least 210 rows and 784 pixels, got {images.shape}.')
    patterns = preprocess_images(images)
    variants = create_variants(patterns, labels)
    all_results = []
    for name, (p, y) in variants.items():
        print(f'Running Variant {name}: rows {1 if name=="A" else 71 if name=="B" else 141}-{70 if name=="A" else 140 if name=="B" else 210}')
        print('Digit distribution:', dict(pd.Series(y).value_counts().sort_index()))
        rows, model = run_experiment(p, y, name)
        all_results.extend(rows)
        show_recall(model, p, y, name, sample_index=0, noise_level=0.30)
    df = pd.DataFrame(all_results)
    df.to_csv('results/final_results.csv', index=False)
    plot_metric(df, 'Exact Recall Accuracy', 'Exact Recall Accuracy (%)', 'Exact Pattern Recall vs Noise', 'recall_accuracy.png')
    plot_metric(df, 'Digit Recall Accuracy', 'Digit Recall Accuracy (%)', 'Digit-Level Recall vs Noise', 'digit_accuracy.png')
    plot_metric(df, 'Normalized Hamming Error', 'Normalized Hamming Error', 'Hamming Error vs Noise', 'hamming_error.png')
    print('\nCompleted. Results saved to results/final_results.csv')
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

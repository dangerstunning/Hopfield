import numpy as np
from hopfield import HopfieldNetwork
from preprocessing import add_noise

NOISE_LEVELS = [0.10, 0.20, 0.30, 0.40]

def hamming_distance(a, b):
    return int(np.sum(a != b))

def identify_digit(recalled, stored_patterns, stored_labels):
    distances = np.sum(stored_patterns != recalled, axis=1)
    return int(stored_labels[np.argmin(distances)])

def run_experiment(patterns, labels, variant_name, seed=42):
    model = HopfieldNetwork(patterns.shape[1], seed=seed)
    model.train(patterns)
    rows = []
    for noise in NOISE_LEVELS:
        exact = digit = total_hamming = total_iterations = 0
        energies = []
        for i, original in enumerate(patterns):
            noisy = add_noise(original, noise, seed=seed + i)
            recalled, iterations, trajectory = model.recall(noisy)
            exact += int(np.array_equal(original, recalled))
            digit += int(identify_digit(recalled, patterns, labels) == labels[i])
            total_hamming += hamming_distance(original, recalled)
            total_iterations += iterations
            energies.append(trajectory[-1])
        n = len(patterns)
        rows.append({
            'Variant': variant_name,
            'Noise': noise,
            'Exact Recall Accuracy': exact / n,
            'Digit Recall Accuracy': digit / n,
            'Average Hamming Distance': total_hamming / n,
            'Normalized Hamming Error': total_hamming / (n * patterns.shape[1]),
            'Average Iterations': total_iterations / n,
            'Average Final Energy': float(np.mean(energies)),
        })
    return rows, model

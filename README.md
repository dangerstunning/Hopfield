# Handwritten Digit Pattern Recall Using Hopfield Networks

## Problem statement
Develop a classical Hopfield Neural Network associative memory that stores handwritten digit patterns from the Kaggle Digit Recognizer dataset and recalls them from corrupted inputs. Compare three independent 70-sample variants: A = rows 1-70, B = rows 71-140, C = rows 141-210.

## Dataset
Kaggle Digit Recognizer: https://www.kaggle.com/competitions/digit-recognizer/data

Download `train.csv` and place it at `data/train.csv`.

## Model
- 784 neurons (28x28 image flattened)
- Bipolar states {-1, +1}
- Hebbian learning
- Zero diagonal / no self-connections
- Asynchronous random-order recall
- Noise levels: 10%, 20%, 30%, 40%

## Metrics
- Exact pattern recall accuracy
- Digit-level recall accuracy
- Average Hamming distance
- Normalized Hamming error
- Average convergence iterations
- Average final Hopfield energy

## Run
```bash
pip install -r requirements.txt
python main.py
```

The generated CSV and plots are written to `results/`.

## Variant definitions
- A: train.csv rows 1-70
- B: train.csv rows 71-140
- C: train.csv rows 141-210

These are independent experiments, each storing exactly 70 patterns.

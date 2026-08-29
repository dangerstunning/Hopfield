import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons, seed=42):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def train(self, patterns):
        patterns = np.asarray(patterns, dtype=np.float32)
        self.weights = (patterns.T @ patterns) / self.n_neurons
        np.fill_diagonal(self.weights, 0.0)

    def update(self, state):
        new_state = state.copy()
        for i in self.rng.permutation(self.n_neurons):
            activation = self.weights[i] @ new_state
            new_state[i] = 1 if activation >= 0 else -1
        return new_state

    def recall(self, pattern, max_iterations=20):
        state = pattern.copy()
        energies = [self.energy(state)]
        for iteration in range(max_iterations):
            old = state.copy()
            state = self.update(state)
            energies.append(self.energy(state))
            if np.array_equal(old, state):
                return state, iteration + 1, energies
        return state, max_iterations, energies

    def energy(self, state):
        return float(-0.5 * (state @ self.weights @ state))

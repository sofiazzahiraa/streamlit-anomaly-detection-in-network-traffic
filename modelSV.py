import numpy as np
from collections import Counter

class PegasosKernelSVM:
    def __init__(self, kernel='rbf', lambda_=0.01, gamma=0.1, n_iters=100):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.gamma = gamma
        self.n_iters = n_iters
        self.alpha = None
        self.X = None
        self.y = None
        self.b = 0

    def _kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            dists = X1_sq - 2 * np.dot(X1, X2.T) + X2_sq
            return np.exp(-self.gamma * dists)
        else:
            raise ValueError("Unsupported kernel")

    def fit(self, X, y):
        n_samples = X.shape[0]
        y = np.where(y == 0, -1, 1)

        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0

        for t in range(1, self.n_iters + 1):
            i = np.random.randint(0, n_samples)
            x_i = X[i].reshape(1, -1)
            y_i = y[i]
            K_i = self._kernel_function(self.X, x_i).flatten()
            margin = y_i * (np.sum(self.alpha * self.y * K_i) + self.b)

            eta = 1 / (self.lambda_ * t)

            if margin < 1:
                self.alpha[i] += eta
                self.b += eta * y_i

    def project(self, X):
        K = self._kernel_function(X, self.X)
        return np.dot(K, self.alpha * self.y) + self.b

    def predict(self, X):
        return np.sign(self.project(X))


class OneVsOneSVM:
    def __init__(self, kernel='rbf', lambda_=0.01, gamma=0.1, n_iters=100, max_samples_per_class=1000):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.gamma = gamma
        self.n_iters = n_iters
        self.max_samples_per_class = max_samples_per_class
        self.models = {}

    def fit(self, X, y):
        self.models = {}
        self.classes_ = np.unique(y)

        for i in range(len(self.classes_)):
            for j in range(i + 1, len(self.classes_)):
                class_i = self.classes_[i]
                class_j = self.classes_[j]

                idx_i = np.where(y == class_i)[0][:self.max_samples_per_class]
                idx_j = np.where(y == class_j)[0][:self.max_samples_per_class]

                idx = np.concatenate([idx_i, idx_j])
                X_pair = X[idx]
                y_pair = y[idx]
                y_pair = np.where(y_pair == class_i, 0, 1)

                model = PegasosKernelSVM(kernel=self.kernel, lambda_=self.lambda_, gamma=self.gamma, n_iters=self.n_iters)
                model.fit(X_pair, y_pair)
                self.models[(class_i, class_j)] = model

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for x in X:
            x = np.array(x).astype(float)
            votes = []
            for (class_i, class_j), model in self.models.items():
                pred = model.predict(x.reshape(1, -1))[0]
                winner = class_i if pred == -1 else class_j
                votes.append(winner)
            final_vote = Counter(votes).most_common(1)[0][0]
            predictions.append(final_vote)

        return np.array(predictions)
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

class Sampling:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def no_sampling(self):
        return self.X_train, self.y_train

    def random_sampling(self, n_samples):
        indices = np.random.choice(len(self.X_train), n_samples, replace=False)
        return self.X_train[indices], self.y_train[indices]

    def optimized_shapley_sampling(self, n_samples, M=1000, epsilon=1e-3, max_iter=100):
        shapley_values = self._compute_shapley_values(M, epsilon)
        
        top_indices = np.argsort(shapley_values)[-n_samples:]
        best_X = self.X_train[top_indices]
        best_y = self.y_train[top_indices]
        best_r2 = self.evaluate_sampling(best_X, best_y)

        for _ in tqdm(range(max_iter), desc="Оптимизация выборки"):
            candidate_indices = top_indices.copy()
            idx_to_remove = np.random.choice(len(candidate_indices))
            idx_to_add = np.random.choice(len(self.X_train))
            candidate_indices[idx_to_remove] = idx_to_add

            candidate_X = self.X_train[candidate_indices]
            candidate_y = self.y_train[candidate_indices]
            candidate_r2 = self.evaluate_sampling(candidate_X, candidate_y)

            if candidate_r2 > best_r2:
                best_X, best_y, best_r2 = candidate_X, candidate_y, candidate_r2
                top_indices = candidate_indices

        return best_X, best_y

    def _compute_shapley_values(self, M, epsilon):
        n_train = self.X_train.shape[0]
        shapley_values = np.zeros(n_train)

        for _ in tqdm(range(M), desc="Вычисление Shapley values"):
            permutation = np.random.permutation(n_train)
            current_coalition_X = np.zeros((0, self.X_train.shape[1]))
            current_coalition_y = np.zeros(0)
            prev_prediction = self.model.predict(self.X_test)[0]

            for i, train_idx in enumerate(permutation):
                current_coalition_X = np.vstack([current_coalition_X, self.X_train[train_idx]])
                current_coalition_y = np.concatenate([current_coalition_y, [self.y_train[train_idx]]])
                self.model.fit(current_coalition_X, current_coalition_y)
                new_prediction = self.model.predict(self.X_test)[0]
                marginal_contribution = new_prediction - prev_prediction
                shapley_values[train_idx] += marginal_contribution

                if abs(marginal_contribution) < epsilon:
                    break

                prev_prediction = new_prediction

        shapley_values /= M
        return shapley_values

    def evaluate_sampling(self, X_sampled, y_sampled):
        self.model.fit(X_sampled, y_sampled)
        y_pred = self.model.predict(self.X_test)
        return r2_score(self.y_test, y_pred)
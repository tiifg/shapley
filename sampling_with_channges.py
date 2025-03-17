import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Ваш класс Sampling (с небольшими изменениями для работы с процентами)
class Sampling:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def no_sampling(self, train_size):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=42)
        return X_train, y_train, X_test, y_test

    def random_sampling(self, train_size):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=42)
        return X_train, y_train, X_test, y_test

    def optimized_shapley_sampling(self, train_size, M=300, epsilon=1e-3, max_iter=80):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=42)
        shapley_values = self._compute_shapley_values(X_train, y_train, X_test, y_test, M, epsilon)
        
        n_samples = int(train_size * len(self.X))
        top_indices = np.argsort(shapley_values)[-n_samples:]
        best_X = X_train[top_indices]
        best_y = y_train[top_indices]
        best_r2 = self.evaluate_sampling(best_X, best_y, X_test, y_test)

        for _ in tqdm(range(max_iter), desc="Оптимизация выборки"):
            candidate_indices = top_indices.copy()
            idx_to_remove = np.random.choice(len(candidate_indices))
            idx_to_add = np.random.choice(len(X_train))
            candidate_indices[idx_to_remove] = idx_to_add

            candidate_X = X_train[candidate_indices]
            candidate_y = y_train[candidate_indices]
            candidate_r2 = self.evaluate_sampling(candidate_X, candidate_y, X_test, y_test)

            if candidate_r2 > best_r2:
                best_X, best_y, best_r2 = candidate_X, candidate_y, candidate_r2
                top_indices = candidate_indices

        return best_X, best_y, X_test, y_test

    def _compute_shapley_values(self, X_train, y_train, X_test, y_test, M, epsilon):
        n_train = X_train.shape[0]
        shapley_values = np.zeros(n_train)

        for _ in tqdm(range(M), desc="Вычисление Shapley values"):
            permutation = np.random.permutation(n_train)
            current_coalition_X = np.zeros((0, X_train.shape[1]))
            current_coalition_y = np.zeros(0)
            prev_prediction = self.model.predict(X_test)[0]

            for i, train_idx in enumerate(permutation):
                current_coalition_X = np.vstack([current_coalition_X, X_train[train_idx]])
                current_coalition_y = np.concatenate([current_coalition_y, [y_train[train_idx]]])
                self.model.fit(current_coalition_X, current_coalition_y)
                new_prediction = self.model.predict(X_test)[0]
                marginal_contribution = new_prediction - prev_prediction
                shapley_values[train_idx] += marginal_contribution

                if abs(marginal_contribution) < epsilon:
                    break

                prev_prediction = new_prediction

        shapley_values /= M
        return shapley_values

    def evaluate_sampling(self, X_sampled, y_sampled, X_test, y_test):
        self.model.fit(X_sampled, y_sampled)
        y_pred = self.model.predict(X_test)
        return r2_score(y_test, y_pred)

    def evaluate_sampling_methods(self, test_sizes):
        results = {'no_sampling': [], 'random_sampling': [], 'optimized_shapley_sampling': []}

        for test_size in tqdm(test_sizes, desc="Оценка методов семплинга"):
            train_size = 1 - test_size

            # No sampling
            X_train, y_train, X_test, y_test = self.no_sampling(train_size)
            r2 = self.evaluate_sampling(X_train, y_train, X_test, y_test)
            results['no_sampling'].append(r2)

            # Random sampling
            X_train, y_train, X_test, y_test = self.random_sampling(train_size)
            r2 = self.evaluate_sampling(X_train, y_train, X_test, y_test)
            results['random_sampling'].append(r2)

            # Optimized Shapley sampling
            X_train, y_train, X_test, y_test = self.optimized_shapley_sampling(train_size)
            r2 = self.evaluate_sampling(X_train, y_train, X_test, y_test)
            results['optimized_shapley_sampling'].append(r2)

        return results

    def plot_results(self, test_sizes, results):
        plt.figure(figsize=(10, 6))
        for method, r2_scores in results.items():
            plt.plot(test_sizes, r2_scores, label=method)

        plt.xlabel('Доля тестовой выборки')
        plt.ylabel('R2 Score')
        plt.title('Сравнение методов семплинга')
        plt.legend()
        plt.grid(True)
        plt.show()

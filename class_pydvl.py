import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from pydvl.value import compute_shapley_values
from pydvl.utils import Dataset, Utility
from pydvl.value.shapley import ShapleyMode, MaxUpdates

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

    def optimized_shapley_sampling(self, n_samples, max_iter=100):
        dataset = Dataset(
            x_train=self.X_train, y_train=self.y_train,  
            x_test=self.X_test, y_test=self.y_test       
        )
        
        utility = Utility(model=self.model, data=dataset)

        shapley_values_tmc = compute_shapley_values(
            utility,
            mode=ShapleyMode.TruncatedMontecarlo,
            done=MaxUpdates(200),
            n_jobs=-1, 
            progress=True
        )
        
        top_indices_tmc = np.argsort(shapley_values_tmc)[-n_samples:]
        X_train_tmc = X_train[top_indices_tmc]
        y_train_tmc = y_train[top_indices_tmc]
        r2_tmc = evaluate_model(model, X_train_tmc, y_train_tmc, X_test, y_test)
        
        
        '''# Выбираем топ-n_samples точек с наибольшими Shapley values
        top_indices = np.argsort(values)[-n_samples:]
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
                top_indices = candidate_indices'''

        return X_train_tmc, y_train_tmc

    def evaluate_sampling(self, X_sampled, y_sampled):
        self.model.fit(X_sampled, y_sampled)
        y_pred = self.model.predict(self.X_test)
        return r2_score(self.y_test, y_pred)

    def evaluate_sampling_methods(self, test_sizes):
        results = {'no_sampling': [], 'random_sampling': [], 'optimized_shapley_sampling': []}

        for test_size in tqdm(test_sizes, desc="Оценка методов семплинга"):
            train_size = 1 - test_size

            # No sampling
            X_train, y_train, X_test, y_test = self.no_sampling()
            r2 = self.evaluate_sampling(X_train, y_train, X_test, y_test)
            results['no_sampling'].append(r2)

            # Random sampling
            X_train, y_train, X_test, y_test = self.random_sampling()
            r2 = self.evaluate_sampling(X_train, y_train, X_test, y_test)
            results['random_sampling'].append(r2)

            # Optimized Shapley sampling
            X_train, y_train, X_test, y_test = self.optimized_shapley_sampling(n_samples=100)
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
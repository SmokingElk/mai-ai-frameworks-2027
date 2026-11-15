"""Градиентный бустинг своими руками на своих деревьях из tree.py.

Регрессия: инициализация средним, далее на каждой итерации обучаем дерево на
псевдо-остатках (антиградиенте MSE) и делаем шаг с learning rate.

Классификация: многоклассовый бустинг через softmax. Для каждого класса держим
свою функцию F_c, считаем вероятности через softmax, остаток = y_c - p_c,
на нём обучаем дерево. Это стандартная схема как в sklearn (multi-logloss).
"""
import numpy as np

from src.tree import DecisionTreeRegressor


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._base = float(y.mean())  # константное начальное приближение
        rng = np.random.default_rng(self.random_state)
        F = np.full(len(y), self._base)
        self._trees = []
        n = len(y)
        for _ in range(self.n_estimators):
            resid = y - F  # антиградиент MSE
            if self.subsample < 1.0:
                # стохастический бустинг: дерево учится на подвыборке
                idx = rng.choice(n, size=max(1, int(self.subsample * n)), replace=False)
            else:
                idx = np.arange(n)
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         random_state=rng.integers(1 << 31))
            tree.fit(X[idx], resid[idx])
            F += self.learning_rate * tree.predict(X)
            self._trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), self._base)
        for tree in self._trees:
            out += self.learning_rate * tree.predict(X)
        return out


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._classes = np.unique(y)
        K = len(self._classes)
        n = len(y)
        Y = np.zeros((n, K))
        Y[np.arange(n), np.searchsorted(self._classes, y)] = 1.0
        rng = np.random.default_rng(self.random_state)
        # начальное приближение: логиты априорных вероятностей классов
        prior = Y.mean(0)
        self._F0 = np.log(np.maximum(prior, 1e-10) / np.maximum(prior[-1], 1e-10))
        F = np.tile(self._F0, (n, 1))
        self._trees = []  # список списков: на итерацию K деревьев (по одному на класс)
        for _ in range(self.n_estimators):
            P = _softmax(F)
            resid = Y - P  # антиградиент кросс-энтропии
            if self.subsample < 1.0:
                idx = rng.choice(n, size=max(1, int(self.subsample * n)), replace=False)
            else:
                idx = np.arange(n)
            round_trees = []
            for c in range(K):
                tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                             min_samples_leaf=self.min_samples_leaf,
                                             random_state=rng.integers(1 << 31))
                tree.fit(X[idx], resid[idx, c])
                F[:, c] += self.learning_rate * tree.predict(X)
                round_trees.append(tree)
            self._trees.append(round_trees)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.tile(self._F0, (len(X), 1))
        for round_trees in self._trees:
            for c, tree in enumerate(round_trees):
                F[:, c] += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        return _softmax(self.decision_function(X))

    def predict(self, X):
        return self._classes[np.argmax(self.decision_function(X), axis=1)]

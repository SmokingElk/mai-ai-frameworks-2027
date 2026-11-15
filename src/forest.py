"""Случайный лес своими руками: бэггинг поверх своих деревьев из tree.py.

Каждое дерево обучается на бутстрап-выборке и смотрит на случайное подмножество
признаков в каждом сплите (max_features).
"""
import numpy as np

from src.tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _n_max_features(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if self.max_features == "all":
            return n_features
        if isinstance(self.max_features, float):
            # доля от всех признаков, как в sklearn
            return max(1, int(round(self.max_features * n_features)))
        return int(self.max_features)

    def _fit(self, X, y, is_classifier):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._is_classifier = is_classifier
        self._classes = np.unique(y) if is_classifier else None
        self._mf = self._n_max_features(X.shape[1])
        rng = np.random.default_rng(self.random_state)
        tree_cls = DecisionTreeClassifier if is_classifier else DecisionTreeRegressor
        self._trees = []
        n = len(y)
        for _ in range(self.n_estimators):
            boot = rng.integers(0, n, size=n)  # бутстрап: выборка с возвращением
            tree = tree_cls(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf,
                            max_features=self._mf,
                            random_state=rng.integers(1 << 31))
            tree.fit(X[boot], y[boot])
            self._trees.append(tree)
        return self


class RandomForestClassifier(RandomForest):
    def fit(self, X, y):
        return self._fit(X, y, True)

    def predict_proba(self, X):
        # усредняем распределения классов по всем деревьям
        return np.mean([t.predict_proba(X) for t in self._trees], axis=0)

    def predict(self, X):
        return self._classes[np.argmax(self.predict_proba(X), axis=1)]


class RandomForestRegressor(RandomForest):
    def fit(self, X, y):
        return self._fit(X, y, False)

    def predict(self, X):
        # усредняем предсказания деревьев
        return np.mean([t.predict(X) for t in self._trees], axis=0)

"""KNN своими руками на numpy. Брутфорс: считаем расстояния до всех обучающих точек.

Для наших размеров данных (тысячи строк) этого достаточно.
"""
import numpy as np


class KNNBase:
    def __init__(self, k=5, weights="uniform", metric="euclidean"):
        self.k = k
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y)
        return self

    def _dist(self, A, B):
        # расстояния от каждой строки A до каждой строки B
        if self.metric == "euclidean":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
            return np.sqrt(np.maximum(sq, 0))
        if self.metric == "manhattan":
            # считаем по одному признаку за раз, чтобы не разворачивать трёхмерный массив
            D = np.zeros((len(A), len(B)))
            for j in range(A.shape[1]):
                D += np.abs(A[:, j][:, None] - B[:, j][None, :])
            return D
        raise ValueError(f"unknown metric: {self.metric}")

    def _neighbors(self, X, batch=512):
        # возвращает (индексы k соседей, расстояния до них) батчами, чтобы не жрать память
        X = np.asarray(X, dtype=float)
        idx_out, dist_out = [], []
        for start in range(0, len(X), batch):
            d = self._dist(X[start:start + batch], self._X)
            part = np.argsort(d, axis=1)[:, :self.k]
            idx_out.append(part)
            dist_out.append(np.take_along_axis(d, part, axis=1))
        return np.vstack(idx_out), np.vstack(dist_out)


class KNNClassifier(KNNBase):
    def __init__(self, k=5, weights="uniform", metric="euclidean"):
        super().__init__(k, weights, metric)

    def fit(self, X, y):
        self._classes = np.unique(y)
        self._class_idx = {c: i for i, c in enumerate(self._classes)}
        self._y_idx = np.array([self._class_idx[v] for v in y])
        return super().fit(X, y)

    def predict_proba(self, X):
        idx, dist = self._neighbors(X)
        n_classes = len(self._classes)
        probs = np.zeros((len(X), n_classes))
        for i in range(len(X)):
            labels = self._y_idx[idx[i]]
            if self.weights == "distance":
                # вес = 1/расстояние (с защитой от нуля)
                w = 1.0 / np.maximum(dist[i], 1e-10)
            else:
                w = np.ones(self.k)
            for j, c in enumerate(self._classes):
                probs[i, j] = w[labels == j].sum()
        return probs / probs.sum(1, keepdims=True)

    def predict(self, X):
        return self._classes[np.argmax(self.predict_proba(X), axis=1)]


class KNNRegressor(KNNBase):
    def __init__(self, k=5, weights="uniform", metric="euclidean"):
        super().__init__(k, weights, metric)

    def predict(self, X):
        idx, dist = self._neighbors(X)
        vals = self._y[idx].astype(float)
        if self.weights == "distance":
            w = 1.0 / np.maximum(dist, 1e-10)
            return (vals * w).sum(1) / w.sum(1)
        return vals.mean(1)

"""Линейная и логистическая регрессия своими руками на numpy.

Линейная: нормальное уравнение (через pinv) или градиентный спуск.
Логистическая: softmax-регрессия (многоклассовая), обучение градиентным спуском.
"""
import numpy as np


def _add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])


class LinearRegressionGD:
    def __init__(self, method="normal", lr=0.1, n_iter=2000):
        self.method = method
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        Xb = _add_bias(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        if self.method == "normal":
            # нормальное уравнение w = (X^T X)^-1 X^T y, pinv на случай плохой обусловленности
            self.w_ = np.linalg.pinv(Xb) @ y
        else:
            n = len(Xb)
            self.w_ = np.zeros(Xb.shape[1])
            for _ in range(self.n_iter):
                grad = 2.0 / n * Xb.T @ (Xb @ self.w_ - y)
                self.w_ -= self.lr * grad
        return self

    def predict(self, X):
        return _add_bias(np.asarray(X, dtype=float)) @ self.w_


class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iter=1000, l2=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)  # для численной стабильности
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def fit(self, X, y):
        Xb = _add_bias(np.asarray(X, dtype=float))
        self._classes = np.unique(y)
        y_idx = np.searchsorted(self._classes, y)
        Y = np.zeros((len(y), len(self._classes)))
        Y[np.arange(len(y)), y_idx] = 1.0
        self.W_ = np.zeros((Xb.shape[1], len(self._classes)))
        n = len(Xb)
        for _ in range(self.n_iter):
            P = self._softmax(Xb @ self.W_)
            G = Xb.T @ (P - Y) / n
            G[1:] += self.l2 * self.W_[1:]  # bias не регуляризуем
            self.W_ -= self.lr * G
        return self

    def predict_proba(self, X):
        Xb = _add_bias(np.asarray(X, dtype=float))
        return self._softmax(Xb @ self.W_)

    def predict(self, X):
        return self._classes[np.argmax(self.predict_proba(X), axis=1)]

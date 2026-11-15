"""Решающее дерево (CART) своими руками на numpy.

Классификация: критерий gini. Регрессия: критерий MSE (минимизируем суммарную
квадратичную ошибку по двум потомкам). Поиск порога эффективный: сортировка
значений признака + кумулятивные суммы, без вложенных циклов по строкам.
"""
import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "dist")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._is_classifier = False

    # ---------- служебное ----------

    def _fit(self, X, y, is_classifier):
        self._is_classifier = is_classifier
        self._rng = np.random.default_rng(self.random_state)
        self._n_features = X.shape[1]
        if is_classifier:
            self._classes = np.unique(y)
            self._class_idx = {c: i for i, c in enumerate(self._classes)}
        self._root = self._build(X, y, depth=0)
        return self

    def _feature_subset(self):
        if self.max_features is None:
            return np.arange(self._n_features)
        k = min(self.max_features, self._n_features)
        return self._rng.choice(self._n_features, size=k, replace=False)

    def _make_leaf(self, node, y):
        node.value = self._leaf_value(y)
        if self._is_classifier:
            counts = np.array([(y == c).sum() for c in self._classes], dtype=float)
            node.dist = counts / counts.sum()

    def _leaf_value(self, y):
        if self._is_classifier:
            counts = np.array([(y == c).sum() for c in self._classes])
            return self._classes[np.argmax(counts)]
        return float(y.mean())

    # ---------- поиск лучшего сплита ----------

    def _best_split(self, X, y):
        n = len(y)
        best = None  # (score, feature, threshold)
        if self._is_classifier:
            # one-hot кодируем классы, чтобы считать количества через cumsum
            K = len(self._classes)
            Y = np.zeros((n, K))
            Y[np.arange(n), np.searchsorted(self._classes, y)] = 1.0
        else:
            y2 = (y ** 2).astype(float)

        for f in self._feature_subset():
            x = X[:, f]
            order = np.argsort(x, kind="stable")
            xs = x[order]
            # позиции, где значение меняется -> валидные пороги
            diff = np.nonzero(xs[1:] != xs[:-1])[0]
            if len(diff) == 0:
                continue
            pos = diff + 1  # левая часть: строки 0..pos-1
            if self._is_classifier:
                cs = np.cumsum(Y[order], axis=0)
                left = cs[pos - 1]              # количества классов слева
                right = cs[-1][None, :] - left  # справа
                nl = pos.astype(float)
                nr = n - nl
                gini_l = 1.0 - (left ** 2).sum(1) / nl ** 2
                gini_r = 1.0 - (right ** 2).sum(1) / nr ** 2
                score = (nl * gini_l + nr * gini_r) / n  # чем меньше, тем лучше
            else:
                ys = y[order].astype(float)
                cs1 = np.cumsum(ys)
                cs2 = np.cumsum(y2[order])
                nl = pos.astype(float)
                nr = n - nl
                sum_l, sum_r = cs1[pos - 1], cs1[-1] - cs1[pos - 1]
                sq_l, sq_r = cs2[pos - 1], cs2[-1] - cs2[pos - 1]
                # SSE = sum(y^2) - sum(y)^2/n для каждой стороны
                sse_l = sq_l - sum_l ** 2 / nl
                sse_r = sq_r - sum_r ** 2 / nr
                score = (sse_l + sse_r) / n
            # фильтр по min_samples_leaf
            ok = (pos >= self.min_samples_leaf) & (n - pos >= self.min_samples_leaf)
            if not ok.any():
                continue
            i = np.argmin(np.where(ok, score, np.inf))
            if best is None or score[i] < best[0] - 1e-12:
                thr = 0.5 * (xs[pos[i] - 1] + xs[pos[i]])
                best = (score[i], f, thr)
        return best

    def _build(self, X, y, depth):
        node = _Node()
        if (len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
                or len(np.unique(y)) == 1):
            self._make_leaf(node, y)
            return node
        split = self._best_split(X, y)
        if split is None:
            self._make_leaf(node, y)
            return node
        _, f, thr = split
        mask = X[:, f] <= thr
        node.feature, node.threshold = f, thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    # ---------- предсказание ----------

    def _predict_one(self, x):
        node = self._root
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _leaf_dist(self, x):
        # спускаемся в лист и возвращаем распределение классов в нём
        node = self._root
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.dist


class DecisionTreeClassifier(DecisionTree):
    def fit(self, X, y):
        return self._fit(np.asarray(X, dtype=float), np.asarray(y), True)

    def predict_proba(self, X):
        # доля классов в листе, в который попал объект
        X = np.asarray(X, dtype=float)
        return np.vstack([self._leaf_dist(x) for x in X])


class DecisionTreeRegressor(DecisionTree):
    def fit(self, X, y):
        return self._fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float), False)

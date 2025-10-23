import numpy as np
from collections import Counter
from scipy.sparse import issparse

class KNN:
    def __init__(self, k=3, metric="euclidean", weights="uniform"):
        """
        k: số neighbors
        metric: 'euclidean' hoặc 'manhattan'
        weights: 'uniform' hoặc 'distance' (trọng số = 1/d)
        """
        self.k = int(k)
        assert metric in ("euclidean", "manhattan")
        assert weights in ("uniform", "distance")
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.asarray(y)
        self.classes_, _ = np.unique(self.y_train, return_inverse=True)
        return self

    def _distance(self, X_train, x):
        """Tính khoảng cách giữa x và toàn bộ X_train, hỗ trợ sparse matrix"""
        if issparse(X_train):
            # chuyển x thành vector dạng (1, n_features)
            if not issparse(x):
                from scipy.sparse import csr_matrix
                x = csr_matrix(x)

            # Euclidean distance
            if self.metric == "euclidean":
                # dùng công thức ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
                X_sq = np.array(X_train.multiply(X_train).sum(axis=1)).flatten()
                x_sq = np.array(x.multiply(x).sum()).flatten()[0]
                cross = X_train.dot(x.T).toarray().flatten()
                dists = np.sqrt(X_sq + x_sq - 2 * cross)
                return dists

            elif self.metric == "manhattan":
                # không có phép trừ trực tiếp hiệu quả -> chuyển tạm dense
                return np.sum(np.abs(X_train.toarray() - x.toarray()), axis=1)
        else:
            # dense numpy array
            if self.metric == "euclidean":
                return np.sqrt(np.sum((X_train - x) ** 2, axis=1))
            elif self.metric == "manhattan":
                return np.sum(np.abs(X_train - x), axis=1)

    def _get_k_neighbors(self, x):
        dists = self._distance(self.X_train, x)
        if self.k >= len(dists):
            idx = np.argsort(dists)
        else:
            idx = np.argpartition(dists, self.k)[:self.k]
        idx = idx[np.argsort(dists[idx])]
        return idx, dists[idx]

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            x = X[i]
            idx, d = self._get_k_neighbors(x)
            neighbor_labels = self.y_train[idx]

            if self.weights == "uniform":
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                preds.append(most_common)
            else:
                weights = np.where(d == 0, 1e9, 1.0 / d)
                label_score = {}
                for lbl, w in zip(neighbor_labels, weights):
                    label_score[lbl] = label_score.get(lbl, 0.0) + w
                best = max(label_score.items(), key=lambda kv: kv[1])[0]
                preds.append(best)
        return np.array(preds)

    def predict_proba(self, X):
        result = []
        for i in range(X.shape[0]):
            x = X[i]
            idx, d = self._get_k_neighbors(x)
            neighbor_labels = self.y_train[idx]
            counts = {c: 0.0 for c in self.classes_}

            if self.weights == "uniform":
                for lbl in neighbor_labels:
                    counts[lbl] += 1.0
            else:
                weights = np.where(d == 0, 1e9, 1.0 / d)
                for lbl, w in zip(neighbor_labels, weights):
                    counts[lbl] += w

            total = sum(counts.values())
            probs = [counts[c] / total if total > 0 else 0.0 for c in self.classes_]
            result.append(probs)
        return np.array(result)

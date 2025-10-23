# knn_from_scratch.py
import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3, metric="euclidean", weights="uniform"):
        """
        k: số neighbors
        metric: 'euclidean' hoặc 'manhattan' (có thể mở rộng)
        weights: 'uniform' hoặc 'distance' (trọng số = 1/d)
        """
        self.k = int(k)
        assert metric in ("euclidean", "manhattan")
        assert weights in ("uniform", "distance")
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        # lưu danh sách các class (dùng cho predict_proba)
        self.classes_, _ = np.unique(self.y_train, return_inverse=True)
        return self

    def _distance(self, a, b):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2, axis=1))
        elif self.metric == "manhattan":
            return np.sum(np.abs(a - b), axis=1)

    def _get_k_neighbors(self, x):
        # tính khoảng cách từ x tới toàn bộ X_train
        dists = self._distance(self.X_train, x)
        # lấy chỉ số k nhỏ nhất (sử dụng argpartition nhanh)
        if self.k >= len(dists):
            idx = np.argsort(dists)
        else:
            idx = np.argpartition(dists, self.k)[: self.k]
            # sắp thứ tự k phần tử đúng theo khoảng cách tăng dần
            idx = idx[np.argsort(dists[idx])]
        return idx, dists[idx]

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for x in X:
            idx, d = self._get_k_neighbors(x)
            neighbor_labels = self.y_train[idx]
            if self.weights == "uniform":
                # đa số bình thường
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                preds.append(most_common)
            else:
                # trọng số 1/d; nếu khoảng cách 0 -> đặt trọng số lớn
                weights = np.where(d == 0, 1e9, 1.0 / d)
                label_score = {}
                for lbl, w in zip(neighbor_labels, weights):
                    label_score[lbl] = label_score.get(lbl, 0.0) + w
                # lấy nhãn có tổng trọng số lớn nhất
                best = max(label_score.items(), key=lambda kv: kv[1])[0]
                preds.append(best)
        return np.array(preds)

    def predict_proba(self, X):
        """Trả về ma trận (n_samples, n_classes) xác suất ước lượng bằng tần suất hoặc weighted freq"""
        X = np.asarray(X)
        result = []
        for x in X:
            idx, d = self._get_k_neighbors(x)
            neighbor_labels = self.y_train[idx]
            # khởi tạo counts
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

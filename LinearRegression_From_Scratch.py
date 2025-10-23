import numpy as np

class LinearRegression_From_Scratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Nếu X là ma trận sparse TF-IDF, chuyển sang dense
        if hasattr(X, "toarray"):
            X = X.toarray()

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Dự đoán tuyến tính
            y_pred = np.dot(X, self.weights) + self.bias

            # Sai số
            error = y_pred - y

            # Gradient của hàm mất mát (MSE)
            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)

            # Cập nhật trọng số
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.dot(X, self.weights) + self.bias

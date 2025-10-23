import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, fit_intercept=True, verbose=False):
        """
        lr: learning rate
        n_iters: số vòng lặp huấn luyện
        fit_intercept: có thêm hệ số bias hay không
        verbose: in log quá trình huấn luyện
        """
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Huấn luyện mô hình Logistic Regression bằng gradient descent.
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        X = self._add_intercept(X)

        # khởi tạo trọng số
        self.theta = np.zeros(X.shape[1])

        for i in range(self.n_iters):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / len(y)
            self.theta -= self.lr * gradient

            if self.verbose and i % 100 == 0:
                loss = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
                print(f"Iter {i}: loss={loss:.4f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        X = self._add_intercept(X)
        z = np.dot(X, self.theta)
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Trả về nhãn 0/1 dựa vào ngưỡng threshold
        """
        return (self.predict_proba(X) >= threshold).astype(int)

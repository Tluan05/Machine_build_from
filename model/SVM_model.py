import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, n_iters=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, y):
        
        # 1. Khởi tạo các tham số
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features) # Bắt đầu với kinh nghiệm bằng 0
        self.b = 0

        # Đảm bảo nhãn là -1 và 1
        y_ = np.where(y <= 0, -1, 1)

        # 2. Vòng lặp huấn luyện (Gradient Descent)
        for _ in range(self.n_iters):
            # Duyệt qua từng điểm dữ liệu để tính toán và cập nhật
            for idx, x_i in enumerate(X):
                
                # 3. Kiểm tra điều kiện Hinge Loss
                # Nếu y * (w.x - b) >= 1, điểm này được phân loại đúng và nằm ngoài lề.
                # Nếu không, điểm này bị lỗi (phân loại sai hoặc nằm trong lề).
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                # 4. Cập nhật w và b dựa trên điều kiện
                if condition:
                    # Nếu phân loại đúng, ta chỉ cập nhật w dựa trên thành phần "tối đa hóa lề".
                    # Đây là đạo hàm của (λ * ||w||^2)
                    gradient_w = 2 * self.lambda_param * self.w
                    self.w -= self.learning_rate * gradient_w
                else:
                    # Nếu phân loại sai, ta cập nhật w và b dựa trên cả hai thành phần.
                    gradient_w = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    gradient_b = -y_[idx]
                    self.w -= self.learning_rate * gradient_w
                    self.b -= self.learning_rate * gradient_b
                    pass

    def predict(self, X):
      
        # Tính toán đầu ra tuyến tính (điểm số)
        linear_output = np.dot(X, self.w) - self.b
        
        # Trả về dấu của điểm số (+1 hoặc -1)
        return np.sign(linear_output)                
    
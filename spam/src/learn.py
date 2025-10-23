# import numpy as np
# import matplotlib.pyplot as plt

# # ===============================
# # 1. Tạo dữ liệu giả lập
# # ===============================
# X = np.array([
#     [0.2, 0.7],
#     [0.3, 0.3],
#     [0.8, 0.5],
#     [0.5, 0.1]
# ])
# y = np.array([1, 0, 1, 0])

# m, n = X.shape
# X = np.c_[np.ones((m, 1)), X]  # thêm cột bias
# w = np.zeros(n + 1)            # khởi tạo weight = 0
# print(X)

import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Tạo dữ liệu giả lập
# ===============================
X = np.array([
    [0.2, 0.7],
    [0.3, 0.3],
    [0.8, 0.5],
    [0.5, 0.1]
])
y = np.array([1, 0, 1, 0])

m, n = X.shape
X = np.c_[np.ones((m, 1)), X]  # thêm cột bias
w = np.zeros(n + 1)            # khởi tạo weight = 0

# ===============================
# 2. Hàm sigmoid, predict và loss
# ===============================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w):
    return(np.dot(X, w))

def loss(y, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return - np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# ===============================
# 3. Gradient Descent với hội tụ
# ===============================
learning_rate = 0.5
tolerance = 1e-6
max_epochs = 100000

previous_loss = float('inf')
loss_history = []        # 📊 Lưu loss qua mỗi vòng
weight_history = []      # 📈 Lưu các trọng số để vẽ quỹ đạo
saved_y_pred = []

for epoch in range(max_epochs):
    y_pred = predict(X, w)
    if (epoch <10):
        print(y_pred)
    saved_y_pred.append(y_pred.copy())
    current_loss = loss(y, y_pred)

    # Lưu lịch sử
    loss_history.append(current_loss)
    weight_history.append(w.copy())

    # Gradient
    gradient = np.dot(X.T, (y_pred - y)) / m

    # Cập nhật weight
    w -= learning_rate * gradient

    # Kiểm tra điều kiện hội tụ
    loss_diff = abs(previous_loss - current_loss)
    grad_norm = np.linalg.norm(gradient)

    if loss_diff < tolerance or grad_norm < tolerance:
        print(f"✅ Hội tụ sau {epoch+1} vòng lặp.")
        break

    previous_loss = current_loss
print(saved_y_pred[-5:])
print("Trọng số cuối cùng:", w)
print("Loss cuối cùng:", current_loss)


# plt.figure(figsize=(8,5))
# plt.plot(saved_y_pred[0], 'o', label='Vòng 1')
# plt.plot(saved_y_pred[len(saved_y_pred)//2], 'x', label=f'Vòng {len(saved_y_pred)//2}')
# plt.plot(saved_y_pred[-1], '.', label=f'Vòng {len(saved_y_pred)}')
# plt.xlabel('Mẫu dữ liệu')
# plt.ylabel('Xác suất dự đoán')
# plt.title('Quá trình dự đoán qua các vòng lặp')
# plt.legend()
# plt.grid(True)
# plt.show()

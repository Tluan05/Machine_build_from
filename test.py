# ==============================
# 📦 1. Import thư viện
# ==============================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# ⚙️ 2. Cài đặt SVM thủ công
# ==============================
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Đảm bảo X là mảng numpy dense
        X = np.array(X)
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            idxs = np.random.permutation(n_samples)
            for idx in idxs:
                x_i = X[idx]
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[idx] * x_i)
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        X = np.array(X)
        approx = np.dot(X, self.w) - self.b
        return np.where(approx >= 0, 1, 0)

# ==============================
# 📂 3. Đọc dữ liệu
# ==============================
data = pd.read_csv("spam_ham.csv", encoding='latin-1')

# Mã hóa nhãn
data['label_num'] = data['label'].map({'Ham': 0, 'Spam': 1})

# ==============================
# 🧠 4. Tách train/test và vector hóa TF-IDF
# ==============================
X = data['text']
y = data['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# ==============================
# 🚀 5. Huấn luyện mô hình
# ==============================
model = SVM(learning_rate=0.001, n_iters=200, lambda_param=0.01)
model.fit(X_train_vec, y_train)

# ==============================
# 🔍 6. Dự đoán và đánh giá
# ==============================
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("===== ĐÁNH GIÁ MÔ HÌNH SVM THỦ CÔNG =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# ==============================
# 📊 7. Ma trận nhầm lẫn
# ==============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Custom SVM")
plt.show()

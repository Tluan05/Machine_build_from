import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- 1. Đọc và chuẩn hóa cột label ---
df = pd.read_csv("src/spam_data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Chuyển nhãn 'spam' -> 1, 'ham' -> 0
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])


# --- 2. Biến đổi text thành vector TF-IDF ---
# Loại bỏ từ xuất hiện quá ít hoặc quá phổ biến để vector gọn hơn
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
)


X_tfidf = vectorizer.fit_transform(df["message"])
y = df["label_encoded"]

# 3. Lấy từ cụ thể dựa trên chỉ số từ output của bạn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Chia tập train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.3, random_state=42, stratify=y
)

print("✅ Train size:", X_train.shape)
print("✅ Test size:", X_test.shape)

# --- 2. Huấn luyện Logistic Regression ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- 3. Dự đoán ---
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --- 4. Đánh giá mô hình ---
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")

# --- 5. Ma trận nhầm lẫn ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- 6. ROC curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# # --- 1. Phân bố số lượng ---
# plt.figure(figsize=(6, 4))
# sns.countplot(x="label", data=df, palette="viridis")
# plt.title("Phân bố số lượng thư Spam vs Ham")
# plt.show()

# # --- 2. Phân bố độ dài tin nhắn ---
# df["length"] = df["message"].apply(len)

# plt.figure(figsize=(10, 5))
# sns.histplot(data=df, x="length", hue="label", bins=50, kde=True, palette="viridis")
# plt.title("Phân bố độ dài tin nhắn theo loại")
# plt.xlabel("Độ dài tin nhắn (số ký tự)")
# plt.ylabel("Số lượng")
# plt.show()

# # --- 3. Top features quan trọng nhất ---
# feature_names = vectorizer.get_feature_names_out()
# coef = model.coef_[0]

# # Lấy top 20 từ có hệ số lớn nhất (spam) và nhỏ nhất (ham)
# top_spam_idx = np.argsort(coef)[-20:]
# top_ham_idx = np.argsort(coef)[:20]

# plt.figure(figsize=(10, 5))
# plt.barh(range(20), coef[top_spam_idx], color="red")
# plt.yticks(range(20), feature_names[top_spam_idx])
# plt.title("Top 20 từ đặc trưng nhất cho Spam")
# plt.xlabel("Trọng số")
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.barh(range(20), coef[top_ham_idx], color="blue")
# plt.yticks(range(20), feature_names[top_ham_idx])
# plt.title("Top 20 từ đặc trưng nhất cho Ham")
# plt.xlabel("Trọng số")
# plt.show()

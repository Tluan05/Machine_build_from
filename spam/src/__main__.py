import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("src/spam_data/spam.csv", encoding="latin-1")
data.info()

data = data[["v1", "v2"]]
data.columns = ["label", "message"]

print(data.head())
print(data["label"].value_counts())


# Chuyển label về 0/1
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

X = data["message"]
y = data["label_num"]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Chuẩn hóa văn bản
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Logistic Regression
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_vec, y_train)

# Dự đoán
y_pred = model.predict(X_test_vec)

# Đánh giá
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

new_emails = [
    "Hi, I have received your email. I will send my assignment on time",
    "Valid 12 hours only.",
]

# Biến đổi theo vectorizer cũ
new_vec = vectorizer.transform(new_emails)

# Dự đoán
pred = model.predict(new_vec)

for email, label in zip(new_emails, pred):
    print(f"Email: {email}\n→ Dự đoán: {'spam' if label == 1 else 'ham'}\n")

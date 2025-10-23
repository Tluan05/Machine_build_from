import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from LinearRegression_From_Scratch import LinearRegression_From_Scratch

# Đọc dữ liệu
df = pd.read_csv("spam_ham.csv")
df.columns = [c.strip().lower() for c in df.columns]

# Chuẩn hóa nhãn + loại bỏ NaN
df['label_num'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})
df = df.dropna(subset=['label_num', 'text'])
df['label_num'] = df['label_num'].astype(int)

# Văn bản
X_text = df['text'].fillna('')
y = df['label_num'].values

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vec = vectorizer.fit_transform(X_text)

# Chia tập
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Huấn luyện Linear Regression (from scratch)
model = LinearRegression_From_Scratch(lr=0.01, n_iters=500)
model.fit(X_train, y_train)

# Dự đoán & đánh giá
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

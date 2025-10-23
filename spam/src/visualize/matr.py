import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# --- 1. Đọc và chuẩn hóa cột label ---
df = pd.read_csv("src/spam_data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Chuyển nhãn 'spam' -> 1, 'ham' -> 0
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])


vectorizer = TfidfVectorizer(
    stop_words="english", max_features=30
)  # ⚠️ giới hạn 30 từ phổ biến nhất
X_tfidf = vectorizer.fit_transform(df["message"])

# Chuyển thành DataFrame
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# ===============================
# 2️⃣ Tính ma trận tương quan
# ===============================
corr_matrix = tfidf_df.corr()

# ===============================
# 3️⃣ Trực quan hóa bằng Plotly
# ===============================
fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="🔍 Ma trận tương quan giữa các từ phổ biến trong tập spam",
)

# ===============================
# 4️⃣ Xuất ra JSON
# ===============================
fig.update_layout(width=900, height=800, title_x=0.5)

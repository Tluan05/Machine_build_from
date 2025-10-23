from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from src.model.knn import KNN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("src/spam_data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]


label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
# -> spam: 1, ham: 0

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
)

X_tfidf = vectorizer.fit_transform(df["message"])
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train)

knn = KNN(k=5, metric="euclidean", weights="distance")
knn.fit(X_train, y_train)

# ===============================
# 5. Dự đoán
# ===============================
y_pred = knn.predict(X_test)

# ===============================
# 6. Đánh giá mô hình
# ===============================
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===============================
# 1. Giảm chiều TF-IDF -> 2D
# ===============================
print("Đang chạy t-SNE (mất vài giây)...")

from sklearn.manifold import TSNE
X_embedded = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    learning_rate="auto",
    init="random"
).fit_transform(X_test.toarray())

# ===============================
# 2. Chuẩn bị dữ liệu
# ===============================
y_true_samples = y_test.to_numpy()
y_pred_samples = knn.predict(X_test)
correct_mask = y_true_samples == y_pred_samples

df_vis = pd.DataFrame({
    "x": X_embedded[:, 0],
    "y": X_embedded[:, 1],
    "True Label": y_true_samples,
    "Predicted": y_pred_samples,
    "Correct": correct_mask
})

# ===============================
# 3. Vẽ scatter tương tác
# ===============================
fig = px.scatter(
    df_vis,
    x="x",
    y="y",
    color="Predicted",
    symbol="Correct",
    symbol_map={True: "circle", False: "x"},
    hover_data={
        "True Label": True,
        "Predicted": True,
        "x": False,
        "y": False
    },
    title="Trực quan cụm dữ liệu test (t-SNE + KNN Prediction)",
)

fig.update_traces(marker=dict(size=8, opacity=0.8))
fig.update_layout(
    template="plotly_white",
    legend_title_text="Predicted Label"
)
fig.show()


# from sklearn.decomposition import PCA


# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_tfidf.toarray())

# df_plot = pd.DataFrame({
#     "PC1": X_pca[:, 0],
#     "PC2": X_pca[:, 1],
#     "Label": df["label"]
# })

# # --- Vẽ biểu đồ bằng Plotly ---
# fig = px.scatter(
#     df_plot,
#     x="PC1",
#     y="PC2",
#     color="Label",
#     title="Phân bố dữ liệu spam vs ham (PCA - 2D)",
#     opacity=0.7
# )
# fig.update_traces(marker=dict(size=5))
# fig.show()


# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, perplexity=50, learning_rate=200, random_state=42)
# X_tsne = tsne.fit_transform(X_tfidf.toarray())

# df_tsne = pd.DataFrame({"x": X_tsne[:, 0], "y": X_tsne[:, 1], "Label": df["label"]})

# fig = px.scatter(
#     df_tsne,
#     x="x",
#     y="y",
#     color="Label",
#     title="Trực quan Spam vs Ham (t-SNE 2D)",
#     opacity=0.7,
# )
# fig.update_traces(marker=dict(size=5))
# fig.show()


# Giả lập corpus nếu bạn chưa có
# corpus = df["message"].tolist()
# vectorizer.fit(corpus)
# X_tfidf = vectorizer.transform(corpus)
# terms = vectorizer.get_feature_names_out()

# terms = vectorizer.get_feature_names_out()
# X_tfidf_dense = X_tfidf.toarray()
# term_sums = np.asarray(X_tfidf.sum(axis=0)).ravel()

# # =============================
# # 1️⃣ Top 30 từ có TF-IDF cao nhất
# # =============================
# top_n = 30
# idx = np.argsort(term_sums)[-top_n:][::-1]
# df_terms = pd.DataFrame({"term": terms[idx], "tfidf_sum": term_sums[idx]})
# fig1 = px.bar(
#     df_terms,
#     x="tfidf_sum",
#     y="term",
#     orientation="h",
#     title=f"Top {top_n} terms by TF-IDF sum",
# )
# fig1.update_layout(yaxis={"categoryorder": "total ascending"})

# # =============================
# # 2️⃣ Phân bố IDF
# # =============================
# idf = vectorizer.idf_
# fig2 = go.Figure()
# fig2.add_trace(go.Histogram(x=idf, nbinsx=50))
# fig2.update_layout(
#     title="Distribution of IDF values", xaxis_title="IDF", yaxis_title="Count"
# )

# # =============================
# # 3️⃣ PCA (TruncatedSVD)
# # =============================
# svd = TruncatedSVD(n_components=2, random_state=42)
# X2 = svd.fit_transform(X_tfidf)
# fig3 = px.scatter(
#     x=X2[:, 0],
#     y=X2[:, 1],
#     title="Truncated SVD (2D) of TF-IDF",
#     labels={"x": "Component 1", "y": "Component 2"},
# )

# # =============================
# # 4️⃣ Cosine similarity heatmap (sample)
# # =============================
# n = min(100, X_tfidf.shape[0])
# idx = np.random.choice(X_tfidf.shape[0], size=n, replace=False)
# X_sample = X_tfidf[idx].toarray()
# sim = cosine_similarity(X_sample)
# fig4 = px.imshow(sim, title="Cosine similarity between sampled documents")

# # =============================
# # 5️⃣ Heatmap nhỏ: Top docs × Top terms
# # =============================
# k_docs, m_terms = 30, 30
# top_terms_idx = np.argsort(term_sums)[-m_terms:][::-1]
# doc_sums = np.asarray(X_tfidf.sum(axis=1)).ravel()
# top_docs_idx = np.argsort(doc_sums)[-k_docs:][::-1]
# sub = X_tfidf[top_docs_idx][:, top_terms_idx].toarray()
# df_heat = pd.DataFrame(
#     sub, index=[f"doc_{i}" for i in top_docs_idx], columns=terms[top_terms_idx]
# )
# fig5 = px.imshow(
#     df_heat,
#     labels=dict(x="term", y="document", color="tf-idf"),
#     aspect="auto",
#     title=f"Heatmap TF-IDF (top {k_docs} docs x top {m_terms} terms)",
# )
# fig5.update_xaxes(tickangle=45)


# from dash import Dash, html, dcc

# app = Dash(__name__)

# app.layout = html.Div(
#     [
#         html.H1("TF-IDF Dashboard"),
#         dcc.Graph(figure=fig1, style={"height": "800px"}),
#         dcc.Graph(figure=fig2, style={"height": "800px"}),
#         dcc.Graph(figure=fig3, style={"height": "800px"}),
#         dcc.Graph(figure=fig4, style={"height": "800px"}),
#         dcc.Graph(figure=fig5, style={"height": "800px"}),
#     ]
# )

# if __name__ == "__main__":
#     app.run(debug=False)


# 3. Lấy từ cụ thể dựa trên chỉ số từ output của bạn

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     roc_curve,
#     auc,
# )
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- 1. Chia tập train/test ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X_tfidf, y, test_size=0.3, random_state=42, stratify=y
# )

# print("✅ Train size:", X_train.shape)
# print("✅ Test size:", X_test.shape)

# # --- 2. Huấn luyện Logistic Regression ---
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # --- 3. Dự đoán ---
# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)[:, 1]

# # --- 4. Đánh giá mô hình ---
# print("\n📊 Classification Report:")
# print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# # Accuracy
# acc = accuracy_score(y_test, y_pred)
# print(f"✅ Accuracy: {acc:.4f}")

# # --- 5. Ma trận nhầm lẫn ---
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(5, 4))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["Ham", "Spam"],
#     yticklabels=["Ham", "Spam"],
# )
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # --- 6. ROC curve ---
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic")
# plt.legend(loc="lower right")
# plt.show()

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

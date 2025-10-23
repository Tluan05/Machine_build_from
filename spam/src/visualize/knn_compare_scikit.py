from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from ..model.knn import KNN
import numpy as np

# tạo bộ dữ liệu 2 lớp dễ nhìn
X, y = make_blobs(
    n_samples=200, centers=3, n_features=2, cluster_std=1.0, random_state=42
)
# chuẩn hoá luôn (RẤT quan trọng với KNN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# model tự viết
my_knn = KNN(k=5, metric="euclidean", weights="distance").fit(X_train, y_train)
preds = my_knn.predict(X_test)
acc = np.mean(preds == y_test)
print("Accuracy (my KNN):", acc)

# sklearn để so sánh
sk = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
sk.fit(X_train, y_train)
print("Accuracy (sklearn):", sk.score(X_test, y_test))

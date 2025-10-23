import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

# ===============================
# 1️⃣ Tạo dữ liệu giả lập
# ===============================
X, y = make_blobs(n_samples=20, centers=2, random_state=42, cluster_std=1.2)
colors = np.array(['blue', 'red'])  # Ham = blue, Spam = red

# Điểm mới (email cần dự đoán)
x_new = np.array([[1, 3]])

# ===============================
# 2️⃣ Chuẩn bị các frame cho từng giá trị k
# ===============================
max_k = 10
frames = []
neighbors = NearestNeighbors(n_neighbors=max_k)
neighbors.fit(X)

# khoảng cách tới tất cả điểm
distances, indices = neighbors.kneighbors(x_new)

for k in range(1, max_k + 1):
    selected_idx = indices[0, :k]

    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=X[:, 0], y=X[:, 1],
                    mode="markers",
                    marker=dict(
                        color=[colors[label] for label in y],
                        size=10,
                        line=dict(width=1, color="black")
                    ),
                    name="Train Data"
                ),
                go.Scatter(
                    x=x_new[:, 0], y=x_new[:, 1],
                    mode="markers+text",
                    marker=dict(symbol="star", color="gold", size=15, line=dict(color="black", width=2)),
                    text=["Email mới"],
                    textposition="top center",
                    name="New Point"
                ),
                go.Scatter(
                    x=X[selected_idx, 0],
                    y=X[selected_idx, 1],
                    mode="markers",
                    marker=dict(size=18, color="rgba(255,255,0,0.4)", line=dict(color="black", width=1)),
                    name=f"{k} Nearest"
                )
            ],
            name=f"k={k}"
        )
    )

# ===============================
# 3️⃣ Layout và animation
# ===============================
fig = go.Figure(
    data=frames[0].data,
    layout=go.Layout(
        xaxis=dict(title="Tần suất từ 1"),
        yaxis=dict(title="Tần suất từ 2"),
        title="🔍 Trực quan KNN: k láng giềng gần nhất thay đổi thế nào",
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "▶️ Play", "method": "animate", "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
            ]
        }]
    ),
    frames=frames
)

fig.update_layout(
    sliders=[{
        "steps": [
            {"method": "animate", "args": [[f"k={k}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
             "label": f"k={k}"} for k in range(1, max_k + 1)
        ],
        "x": 0.1, "y": -0.2, "len": 0.8
    }]
)

fig.show()

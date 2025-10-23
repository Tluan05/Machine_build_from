import numpy as np
import plotly.graph_objects as go

# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.linspace(-10, 10, 500)
# Tạo hệ số w từ -5 đến 5 với bước 0.05
w_values = np.round(np.arange(-5, 5.05, 0.05), 2)

frames = []
for w in w_values:
    z = w * x
    y = sigmoid(z)
    frames.append(go.Frame(
        data=[go.Scatter(x=x, y=y, mode='lines')],
        name=str(w)
    ))

# Biểu đồ ban đầu
init_y = sigmoid(0 * x)
fig = go.Figure(
    data=[go.Scatter(x=x, y=init_y, mode='lines')],
    layout=go.Layout(
        title="Sigmoid với slider hệ số w (bước 0.05)",
        xaxis=dict(title='z'),
        yaxis=dict(title='σ(z)', range=[-0.05, 1.05]),
        sliders=[{
            "steps": [
                {
                    "args": [[str(w)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate"
                    }],
                    "label": f"{w:.2f}",
                    "method": "animate"
                } for w in w_values
            ],
            "currentvalue": {"prefix": "w = "}
        }]
    ),
    frames=frames
)

fig.show()

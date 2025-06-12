import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat

# 1. Load prediction results
mat = loadmat('data/prediction_result.mat')
y_true = mat['YVal_numeric'].flatten()
y_pred = mat['YPred_numeric'].flatten()
val_indices = mat['val_indices'].flatten()

# 2. Load CSV
df = pd.read_csv('data/btc_15min_with_indicators_and_labels.csv', parse_dates=['datetime'])

# 3. Filter using indices from .mat file
df_valid = df.loc[val_indices].copy().reset_index(drop=True)

# ✅ Check alignment
assert len(df_valid) == len(y_pred), "Mismatch between CSV and prediction_result.mat"

# 4. Sort by datetime to avoid messy plot
df_valid = df_valid.sort_values('datetime').reset_index(drop=True)

# 4-1. Fix label: convert unexpected values (like 2) to 0
y_pred = np.where(y_pred == 2, 0, y_pred)

# 5. Add columns
df_valid['True Label'] = y_true
df_valid['Predicted Label'] = y_pred

df_valid['Predicted Label'] = df_valid['Predicted Label'].astype(int)
up = df_valid[df_valid['Predicted Label'] == 1]
down = df_valid[df_valid['Predicted Label'] == 0]
print(f"✅ up: {len(up)}, down: {len(down)}")

unique, counts = np.unique(y_pred, return_counts=True)
print("📊 Raw Y_Pred distribution:", dict(zip(unique, counts)))

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_valid['datetime'],
    y=df_valid['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='gray', width=1)
))

fig.add_trace(go.Scatter(
    x=up['datetime'],
    y=up['Close'],
    mode='markers',
    name='Predicted Up',
    marker=dict(color='red', symbol='triangle-up', size=7)
))

if not down.empty:
    fig.add_trace(go.Scatter(
        x=down['datetime'],
        y=down['Close'],
        mode='markers',
        name='Predicted Down',
        marker=dict(color='blue', symbol='triangle-down', size=7)
    ))

fig.update_layout(
    title='📈 BTC Close Price with Predicted Labels',
    xaxis_title='Time',
    yaxis_title='Price',
    legend_title='Legend',
    template='plotly_white',
    height=600
)

# Save the figure as an HTML file
fig.write_html("visualization/btc_prediction_visualization.html")

fig.show()
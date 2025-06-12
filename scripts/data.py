import pandas as pd
import numpy as np
from ta import add_all_ta_features
import matplotlib.pyplot as plt
from scipy.io import savemat

# 1. Load preprocessed 15-min data with indicators and labels
df = pd.read_csv("data/btc_15min_with_indicators_and_labels.csv", parse_dates=["datetime"])
df.set_index("datetime", inplace=True)

# 2. Filter only rows with usable labels
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# 3. Select relevant columns for training input (60 timestep 시퀀스 기준)
selected_columns = [
    "Open", "High", "Low", "Close", "Volume",
    "rsi_golden_cross", "macd_cross_up", "bb_contraction",
    "adx_strong", "bb_break_upper"
]

features = df[selected_columns]
labels = df["label"]
X, y, timestamps = [], [], []

sequence_length = 60

for i in range(len(df) - sequence_length):
    X_seq = features.iloc[i : i + sequence_length].values
    label = labels.iloc[i + sequence_length - 1]
    ts = df.index[i + sequence_length - 1]

    if np.isnan(X_seq).any() or np.isnan(label):
        continue

    X.append(X_seq)
    y.append(label)
    timestamps.append(ts)

print("✅ Final usable samples:", len(y))

X = np.array(X)
y = np.array(y)

# Normalize
mu = np.mean(X, axis=(0, 1))
sigma = np.std(X, axis=(0, 1))
sigma[sigma == 0] = 1
X_norm = (X - mu) / sigma

# Plot class distribution
counts = [np.sum(y == 0), np.sum(y == 1)]
plt.bar(["하락(0)", "상승(1)"], counts)
plt.title("📊 라벨 분포")
plt.ylabel("샘플 수")
plt.show()

# Save .mat file
savemat("btc_features.mat", {
    "X": X_norm,
    "y": y.reshape(-1, 1),
    "mu": mu,
    "sigma": sigma,
    "timestamps": np.array(timestamps, dtype="O")
})

print("📁 Saved training data to btc_features.mat")
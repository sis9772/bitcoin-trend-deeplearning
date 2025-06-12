import numpy as np
from scipy.io import loadmat
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 데이터 불러오기ㅋ
data = loadmat('data/btc_features.mat')
X = data['X']    # (N, T, F)
y = data['y'].flatten()

# 시퀀스를 2D로 펼치기 (N x (T*F))
X_flat = X.reshape(X.shape[0], -1)

# t-SNE 적용
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_flat)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], label='하락(0)', alpha=0.5)
plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], label='상승(1)', alpha=0.5)
plt.legend()
plt.title('📉 t-SNE: 시퀀스 군집 시각화')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)
plt.show()
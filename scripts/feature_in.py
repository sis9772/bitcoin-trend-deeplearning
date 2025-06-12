import scipy.io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# .mat 파일 로드
data = scipy.io.loadmat('btc_features.mat')
X = data['X']  # (N, T, F)
y = data['y'].flatten()

# 시퀀스 평균 pooling
X_avg = X.mean(axis=1)  # (N, F)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_avg, y, test_size=0.2, random_state=42)

# 랜덤포레스트로 중요도 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 중요도 출력
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("📊 상위 중요 피처:")
for i in sorted_idx[:10]:
    print(f"  Feature {i} - Importance: {importances[i]:.4f}")

# 중요도 기반 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.bar(range(len(importances)), importances)
plt.title("📌 Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
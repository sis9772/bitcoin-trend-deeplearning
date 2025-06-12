import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 파일 불러오기
features = scipy.io.loadmat("data/btc_features.mat")
results = scipy.io.loadmat("data/prediction_result.mat")

# 정답과 예측
y_all = features["y"].flatten()
n_total = len(y_all)
val_start = int(n_total * 0.8)
y_true = y_all[val_start:val_start + len(results["YPred_numeric"].flatten())]
y_pred = results["YPred_numeric"].flatten()
y_true = y_true.astype(str)
y_pred = y_pred.astype(str)

# 🧾 성능 지표 출력
print("📄 Classification Report")
print(classification_report(y_true, y_pred, labels=["0", "1"], target_names=["Down", "Up"]))

# 📊 혼동행렬
cm = confusion_matrix(y_true, y_pred, labels=["0", "1"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
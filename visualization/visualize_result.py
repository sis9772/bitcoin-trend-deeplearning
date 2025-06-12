

import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load prediction result from MATLAB
data = scipy.io.loadmat("prediction_result.mat")
y_true = data['YVal_numeric'].flatten()
y_pred = data['YPred_numeric'].flatten()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Python Visualization)")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred))
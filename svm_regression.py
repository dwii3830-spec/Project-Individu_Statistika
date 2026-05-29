# ============================================================
#   SVM REGRESSION (SVR) - Prediksi Perkembangan Diabetes
#   Dataset: Diabetes (bawaan scikit-learn, data medis nyata)
#   Library: scikit-learn
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)
y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train_sc, y_train_sc)

y_pred_sc = model.predict(X_test_sc)
y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

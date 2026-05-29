# ============================================================
#   SVM REGRESSION (SVR) - Prediksi Perkembangan Diabetes
#   Dataset: Diabetes (bawaan scikit-learn, data medis nyata)
#   Library: scikit-learn
# ============================================================

# --- 1. Import Library ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# 2. LOAD DATA NYATA
# ============================================================
print("=" * 50)
print("LOADING DATA...")
print("=" * 50)

data = load_diabetes()
X = data.data    # 10 fitur medis (usia, BMI, tekanan darah, dll)
y = data.target  # Target: skor perkembangan penyakit (1 tahun kemudian)

print(f"Jumlah data  : {X.shape[0]} pasien")
print(f"Jumlah fitur : {X.shape[1]} fitur medis")
print(f"Nama fitur   : {data.feature_names}")
print(f"Target skor  : min={y.min():.0f}, max={y.max():.0f}, rata2={y.mean():.1f}")

# ============================================================
# 3. SPLIT DATA: TRAINING & TESTING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,   # 20% untuk testing
    random_state=42
)

print(f"\nData Training : {len(X_train)} pasien")
print(f"Data Testing  : {len(X_test)} pasien")

# ============================================================
# 4. NORMALISASI FITUR (WAJIB untuk SVR!)
# ============================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

print("\nNormalisasi selesai (StandardScaler)")

# ============================================================
# 5. BUAT MODEL SVR & LATIH
# ============================================================
print("\n" + "=" * 50)
print("MELATIH MODEL SVR...")
print("=" * 50)

# kernel='rbf' -> Radial Basis Function (paling umum untuk non-linear)
# C=1.0        -> Penalti error (makin besar = makin ketat fit ke data)
# epsilon=0.1  -> Toleransi / lebar tube SVR
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train_sc, y_train_sc)

print("Model selesai dilatih!")
print(f"Jumlah Support Vectors: {len(model.support_vectors_)}")

# ============================================================
# 6. PREDIKSI & EVALUASI
# ============================================================
print("\n" + "=" * 50)
print("EVALUASI MODEL")
print("=" * 50)

y_pred_sc = model.predict(X_test_sc)
y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}  (rata-rata error dalam satuan skor)")
print(f"MAE  : {mae:.2f}  (rata-rata absolute error)")
print(f"R²   : {r2:.4f}  (1.0 = sempurna, 0 = jelek)")

print()
if r2 >= 0.5:
    print(f"  → R²={r2:.2f}: Model cukup baik!")
else:
    print(f"  → R²={r2:.2f}: Lumayan, bisa ditingkatkan dengan tuning C/epsilon/kernel")

# ============================================================
# 7. VISUALISASI
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("SVR - Prediksi Skor Perkembangan Diabetes", fontsize=13, fontweight='bold')

# Plot 1: Actual vs Predicted
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='white', linewidths=0.3, s=50)
lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax1.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Prediksi Sempurna')
ax1.set_xlabel("Skor Aktual")
ax1.set_ylabel("Skor Prediksi")
ax1.set_title(f"Actual vs Predicted  (R²={r2:.3f})")
ax1.legend(); ax1.grid(True, alpha=0.3)

# Plot 2: Residual
ax2 = axes[1]
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6, color='tomato', edgecolors='white', linewidths=0.3, s=50)
ax2.axhline(0, color='black', linestyle='--', lw=2)
ax2.set_xlabel("Skor Prediksi")
ax2.set_ylabel("Residual (Aktual - Prediksi)")
ax2.set_title("Residual Plot\n(Idealnya acak di sekitar 0)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/svr_hasil.png", dpi=150, bbox_inches='tight')
print("\nGrafik disimpan: svr_hasil.png")
plt.close()

# ============================================================
# 8. CONTOH PREDIKSI 3 PASIEN
# ============================================================
print("\n" + "=" * 50)
print("CONTOH PREDIKSI 3 PASIEN (dari data test)")
print("=" * 50)
print(f"{'No':<4} {'Aktual':>8} {'Prediksi':>10} {'Error':>8}")
print("-" * 35)
for i in range(3):
    aktual  = y_test[i]
    prediksi = y_pred[i]
    err     = abs(aktual - prediksi)
    print(f"{i+1:<4} {aktual:>8.1f} {prediksi:>10.1f} {err:>8.1f}")

print("\nSelesai!")

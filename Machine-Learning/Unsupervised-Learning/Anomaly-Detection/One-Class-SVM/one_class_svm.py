from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
FIGURES_DIR = BASE_DIR / 'figures'
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(BASE_DIR)

def save_figure(filename):
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close('all')


# 1. Kütüphanelerin İçe Aktarılması

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
print(' Tüm kütüphaneler başarıyla yüklendi.')


# 2. Veri Üretimi

np.random.seed(42)
X_train, _ = make_blobs(n_samples=300, centers=[[0, 0]], cluster_std=0.5, random_state=42)
X_test_normal, _ = make_blobs(n_samples=80, centers=[[0, 0]], cluster_std=0.5, random_state=10)
X_test_anomaly = np.random.uniform(low=-4, high=4, size=(20, 2))
X_test = np.vstack([X_test_normal, X_test_anomaly])
y_test_true = np.array([1] * len(X_test_normal) + [-1] * len(X_test_anomaly))
print(f'Eğitim seti boyutu  : {X_train.shape}  → {X_train.shape[0]} normal nokta')
print(f'Test seti boyutu    : {X_test.shape}   → {len(X_test_normal)} normal + {len(X_test_anomaly)} anormal')


# 3. Özellik Ölçekleme (Feature Scaling)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f'Eğitim verisi — Ortalama: {X_train_scaled.mean(axis=0).round(4)}')
print(f'Eğitim verisi — Std Dev : {X_train_scaled.std(axis=0).round(4)}')
print('\n Ölçekleme tamamlandı (ortalama≈0, std≈1 olmalı)')


# 4. Model Tanımı ve Hiperparametreler

model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
print(' Model oluşturuldu:')
print(model)


# 5. Modeli Eğitme

model.fit(X_train_scaled)
train_predictions = model.predict(X_train_scaled)
n_inliers_train = (train_predictions == 1).sum()
n_outliers_train = (train_predictions == -1).sum()
print(f' Model eğitildi — {X_train_scaled.shape[0]} normal nokta kullanıldı')
print(f'\nEğitim verisi üzerindeki sonuçlar:')
print(f'  Normal  (+1): {n_inliers_train}')
print(f'  Anormal (-1): {n_outliers_train}  ← nu={model.nu} nedeniyle beklenen ≈ {int(len(X_train) * model.nu)} nokta')


# 6. Test Verisi Üzerinde Tahmin

y_pred = model.predict(X_test_scaled)
scores = model.decision_function(X_test_scaled)
correct_mask = y_pred == y_test_true
accuracy = correct_mask.mean() * 100
TP = ((y_pred == 1) & (y_test_true == 1)).sum()
TN = ((y_pred == -1) & (y_test_true == -1)).sum()
FP = ((y_pred == 1) & (y_test_true == -1)).sum()
FN = ((y_pred == -1) & (y_test_true == 1)).sum()
print(' TEST SONUÇLARI')
print('=' * 40)
print(f'Genel Doğruluk         : %{accuracy:.1f}')
print()
print(f'Gerçek Pozitif  (TP)  : {TP:3d}   Normal,  doğru tahmin')
print(f'Gerçek Negatif  (TN)  : {TN:3d}   Anormal, doğru tahmin')
print(f'Yanlış Pozitif  (FP)  : {FP:3d}   Anormal, normal diye geçti')
print(f'Yanlış Negatif  (FN)  : {FN:3d}   Normal,  anormal diye işaretlendi')
print()
print(f'Karar skoru — Min: {scores.min():.3f} | Maks: {scores.max():.3f} | Ort: {scores.mean():.3f}')


# 7. Karar Sınırı Görselleştirmesi

xx, yy = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid_points)
Z = model.decision_function(grid_scaled)
Z = Z.reshape(xx.shape)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('One-Class SVM — Anomali Tespiti', fontsize=15, fontweight='bold', y=1.02)
ax1 = axes[0]
cf = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.4)
plt.colorbar(cf, ax=ax1, label='Karar Skoru (+ normal, − anomali)')
ax1.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkgreen', linestyles='--')
ax1.scatter(X_train[:, 0], X_train[:, 1], c='steelblue', s=15, alpha=0.5, label=f'Eğitim (normal, n={len(X_train)})')
ax1.set_title('Eğitim Verisi ve Öğrenilen Karar Sınırı', fontsize=12)
ax1.set_xlabel('Özellik 1')
ax1.set_ylabel('Özellik 2')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax2 = axes[1]
ax2.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.3)
ax2.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkgreen', linestyles='--')
mask_TP = (y_pred == 1) & (y_test_true == 1)
ax2.scatter(X_test[mask_TP, 0], X_test[mask_TP, 1], c='blue', marker='o', s=40, label=f'TP Normal ({mask_TP.sum()})', zorder=5)
mask_TN = (y_pred == -1) & (y_test_true == -1)
ax2.scatter(X_test[mask_TN, 0], X_test[mask_TN, 1], c='red', marker='X', s=80, label=f'TN Anormal ({mask_TN.sum()})', zorder=5)
mask_FP = (y_pred == 1) & (y_test_true == -1)
ax2.scatter(X_test[mask_FP, 0], X_test[mask_FP, 1], c='orange', marker='X', s=120, edgecolors='black', label=f'FP Kaçırılan Anomali ({mask_FP.sum()})', zorder=6)
mask_FN = (y_pred == -1) & (y_test_true == 1)
ax2.scatter(X_test[mask_FN, 0], X_test[mask_FN, 1], c='purple', marker='o', s=80, edgecolors='black', label=f'FN Yanlış Alarm ({mask_FN.sum()})', zorder=6)
ax2.set_title(f'Test Sonuçları — Doğruluk: %{accuracy:.1f}', fontsize=12)
ax2.set_xlabel('Özellik 1')
ax2.set_ylabel('Özellik 2')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'one_class_svm_results.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' Grafik kaydedildi: one_class_svm_sonuclar.png')


# 8. Nu Parametresinin Etkisi

nu_values = [0.01, 0.05, 0.1, 0.2, 0.3]
results = []
for nu in nu_values:
    m = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    m.fit(X_train_scaled)
    preds = m.predict(X_test_scaled)
    tp = ((preds == 1) & (y_test_true == 1)).sum()
    tn = ((preds == -1) & (y_test_true == -1)).sum()
    fp = ((preds == 1) & (y_test_true == -1)).sum()
    fn = ((preds == -1) & (y_test_true == 1)).sum()
    acc = (preds == y_test_true).mean() * 100
    results.append({'nu': nu, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Acc': acc})
print('┌────────┬──────┬──────┬──────┬──────┬──────────┐')
print('│   nu   │  TP  │  TN  │  FP  │  FN  │ Doğruluk │')
print('├────────┼──────┼──────┼──────┼──────┼──────────┤')
for r in results:
    print(f"│ {r['nu']:.2f}   │ {r['TP']:4d} │ {r['TN']:4d} │ {r['FP']:4d} │ {r['FN']:4d} │  %{r['Acc']:5.1f}   │")
print('└────────┴──────┴──────┴──────┴──────┴──────────┘')
print('\nTP=Doğru Normal | TN=Doğru Anormal | FP=Kaçan Anomali | FN=Yanlış Alarm')


# 9. Nu Karşılaştırma Grafiği

fig, axes = plt.subplots(1, len(nu_values), figsize=(20, 4))
fig.suptitle('Nu Değerinin Karar Sınırına Etkisi', fontsize=14, fontweight='bold')
for i, nu in enumerate(nu_values):
    ax = axes[i]
    m = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    m.fit(X_train_scaled)
    Z_nu = m.decision_function(grid_scaled).reshape(xx.shape)
    ax.contourf(xx, yy, Z_nu, levels=15, cmap='RdYlGn', alpha=0.4)
    ax.contour(xx, yy, Z_nu, levels=[0], linewidths=2.5, colors='darkgreen', linestyles='--')
    ax.scatter(X_train[:, 0], X_train[:, 1], c='steelblue', s=8, alpha=0.4, label='Eğitim')
    ax.scatter(X_test_anomaly[:, 0], X_test_anomaly[:, 1], c='red', marker='X', s=60, alpha=0.8, label='Anomali')
    acc = results[i]['Acc']
    ax.set_title(f'nu={nu}\nDoğruluk: %{acc:.1f}', fontsize=10, fontweight='bold')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'nu_comparison.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' Grafik kaydedildi: nu_karsilastirma.png')


# 10. Özet ve Sonuçlar

print('=' * 50)
print('     ONE-CLASS SVM — SONUÇ ÖZETİ')
print('=' * 50)
print(f'  Eğitim seti    : {len(X_train)} normal nokta')
print(f'  Test seti      : {len(X_test)} nokta ({len(X_test_normal)} normal + {len(X_test_anomaly)} anormal)')
print(f'  Kernel         : RBF')
print(f'  Nu (seçilen)   : {model.nu}')
print(f'  Genel Doğruluk : %{accuracy:.1f}')
print(f'  Tespit Edilen  : {TN}/{len(X_test_anomaly)} anomali (%{TN / len(X_test_anomaly) * 100:.0f})')
print('=' * 50)
print('\n çalışma başarıyla tamamlandı!')

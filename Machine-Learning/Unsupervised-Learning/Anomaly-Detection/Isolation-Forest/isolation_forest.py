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
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(' Tüm kütüphaneler başarıyla yüklendi.')


# 2. Veri Üretimi

np.random.seed(42)
X_cluster1, _ = make_blobs(n_samples=200, centers=[[2, 2]], cluster_std=0.6, random_state=42)
X_cluster2, _ = make_blobs(n_samples=150, centers=[[-2, -2]], cluster_std=0.5, random_state=42)
X_train = np.vstack([X_cluster1, X_cluster2])
X_test_n1, _ = make_blobs(n_samples=60, centers=[[2, 2]], cluster_std=0.6, random_state=7)
X_test_n2, _ = make_blobs(n_samples=40, centers=[[-2, -2]], cluster_std=0.5, random_state=7)
X_test_normal = np.vstack([X_test_n1, X_test_n2])
X_test_anomaly = np.random.uniform(low=-6, high=6, size=(30, 2))
X_test = np.vstack([X_test_normal, X_test_anomaly])
y_true = np.array([1] * len(X_test_normal) + [-1] * len(X_test_anomaly))
print(f'Eğitim seti    : {X_train.shape}  → {len(X_train)} normal nokta (2 küme)')
print(f'Test  seti     : {X_test.shape}  → {len(X_test_normal)} normal + {len(X_test_anomaly)} anormal')
print(f'Anomali oranı  : %{len(X_test_anomaly) / len(X_test) * 100:.1f}')


# 3. Ham Veri Görselleştirmesi

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Ham Veri — Eğitim ve Test Setleri', fontsize=13, fontweight='bold')
ax = axes[0]
ax.scatter(X_cluster1[:, 0], X_cluster1[:, 1], c='steelblue', s=20, alpha=0.6, label='Normal Küme 1 (merkez: 2,2)')
ax.scatter(X_cluster2[:, 0], X_cluster2[:, 1], c='teal', s=20, alpha=0.6, label='Normal Küme 2 (merkez: -2,-2)')
ax.set_title(f'Eğitim Seti — {len(X_train)} Normal Nokta')
ax.set_xlabel('Özellik 1 (örn. sıcaklık)')
ax.set_ylabel('Özellik 2 (örn. titreşim)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax = axes[1]
ax.scatter(X_test_normal[:, 0], X_test_normal[:, 1], c='steelblue', s=25, alpha=0.6, label=f'Normal ({len(X_test_normal)})')
ax.scatter(X_test_anomaly[:, 0], X_test_anomaly[:, 1], c='red', marker='X', s=80, alpha=0.8, label=f'Anormal ({len(X_test_anomaly)})')
ax.set_title(f'Test Seti — {len(X_test)} Nokta')
ax.set_xlabel('Özellik 1')
ax.set_ylabel('Özellik 2')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'raw_data.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' ham_veri.png kaydedildi')


# 4. Özellik Ölçekleme

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f'Eğitim — Ortalama : {X_train_scaled.mean(axis=0).round(6)}  (≈ 0 olmalı)')
print(f'Eğitim — Std Dev  : {X_train_scaled.std(axis=0).round(6)}  (≈ 1 olmalı)')
print('\n Ölçekleme tamamlandı.')


# 5. Isolation Forest — Algoritma Mantığı

model = IsolationForest(n_estimators=200, contamination=0.1, max_samples='auto', max_features=1.0, random_state=42)
print(' Isolation Forest modeli oluşturuldu:')
print(f'   Ağaç sayısı      : {model.n_estimators}')
print(f'   Contamination    : {model.contamination}')
print(f'   Max samples      : {model.max_samples}')


# 6. Modeli Eğitme

model.fit(X_train_scaled)
train_preds = model.predict(X_train_scaled)
n_inliers = (train_preds == 1).sum()
n_outliers = (train_preds == -1).sum()
print(f' Model eğitildi — {len(X_train_scaled)} örnekle, {model.n_estimators} ağaç')
print(f'\nEğitim verisi üzerindeki sonuçlar:')
print(f'  Normal  (+1): {n_inliers}')
print(f'  Anormal (-1): {n_outliers}  ← contamination={model.contamination} → beklenen ≈ {int(len(X_train) * model.contamination)}')


# 7. Test Verisi Üzerinde Tahmin ve Anomali Skorları

y_pred = model.predict(X_test_scaled)
anomaly_scores = model.score_samples(X_test_scaled)
anomaly_scores_pos = -anomaly_scores
TP = ((y_pred == 1) & (y_true == 1)).sum()
TN = ((y_pred == -1) & (y_true == -1)).sum()
FP = ((y_pred == 1) & (y_true == -1)).sum()
FN = ((y_pred == -1) & (y_true == 1)).sum()
accuracy = (y_pred == y_true).mean() * 100
precision = TN / (TN + FP) if TN + FP > 0 else 0
recall = TN / (TN + FN) if TN + FN > 0 else 0
y_true_bin = (y_true == -1).astype(int)
auc = roc_auc_score(y_true_bin, anomaly_scores_pos)
print(' TEST SONUÇLARI')
print('=' * 45)
print(f'Genel Doğruluk      : %{accuracy:.1f}')
print(f'Anomali Precision   : %{precision * 100:.1f}  (tespit edilen anomalilerin doğruluğu)')
print(f'Anomali Recall      : %{recall * 100:.1f}  (gerçek anomalileri yakalama oranı)')
print(f'ROC-AUC Skoru       : {auc:.4f}  (1.0 = mükemmel)')
print()
print(f'TP={TP} | TN={TN} | FP={FP} | FN={FN}')
print()
print(f'Anomali skoru — Min: {anomaly_scores_pos.min():.4f} | Maks: {anomaly_scores_pos.max():.4f} | Ort: {anomaly_scores_pos.mean():.4f}')


# 8. Anomali Skorlarının Dağılımı

scores_normal = anomaly_scores_pos[y_true == 1]
scores_anomaly = anomaly_scores_pos[y_true == -1]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Anomali Skorlarının Analizi', fontsize=13, fontweight='bold')
ax = axes[0]
ax.hist(scores_normal, bins=25, alpha=0.6, color='steelblue', label=f'Normal (n={len(scores_normal)})', density=True)
ax.hist(scores_anomaly, bins=25, alpha=0.6, color='red', label=f'Anormal (n={len(scores_anomaly)})', density=True)
threshold = np.percentile(anomaly_scores_pos, (1 - model.contamination) * 100)
ax.axvline(x=threshold, color='orange', linestyle='--', linewidth=2, label=f'Karar Eşiği = {threshold:.3f}')
ax.set_xlabel('Anomali Skoru (yüksek = daha anormal)')
ax.set_ylabel('Yoğunluk')
ax.set_title('Skor Dağılımı: Normal vs Anormal')
ax.legend()
ax.grid(True, alpha=0.3)
ax = axes[1]
bp = ax.boxplot([scores_normal, scores_anomaly], labels=['Normal', 'Anormal'], patch_artist=True, notch=False, showfliers=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor('salmon')
bp['boxes'][1].set_alpha(0.6)
ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Karar Eşiği = {threshold:.3f}')
ax.set_ylabel('Anomali Skoru')
ax.set_title('Kutu Grafiği: Skor Karşılaştırması')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'score_distribution.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(f' skor_dagilimi.png kaydedildi')
print(f'\nOrtalama Skor — Normal: {scores_normal.mean():.4f} | Anormal: {scores_anomaly.mean():.4f}')
print(f'Karar eşiği: {threshold:.4f}  →  bu eşiğin üstü anormal sayılır')


# 9. Karar Sınırı ve Tespit Sonuçları

xx, yy = np.meshgrid(np.linspace(-7, 7, 300), np.linspace(-7, 7, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid_points)
Z = model.score_samples(grid_scaled)
Z = Z.reshape(xx.shape)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Isolation Forest — Karar Sınırı ve Test Sonuçları', fontsize=14, fontweight='bold')
ax1 = axes[0]
cf = ax1.contourf(xx, yy, Z, levels=25, cmap='RdYlGn', alpha=0.5)
plt.colorbar(cf, ax=ax1, label='Anomali Skoru (yüksek = normal)')
decision_threshold = np.percentile(model.score_samples(X_train_scaled), model.contamination * 100)
ax1.contour(xx, yy, Z, levels=[decision_threshold], linewidths=2, colors='black', linestyles='--')
ax1.scatter(X_train[:, 0], X_train[:, 1], c='steelblue', s=12, alpha=0.4, label=f'Eğitim normal ({len(X_train)})')
ax1.set_title('Öğrenilen Karar Yüzeyi')
ax1.set_xlabel('Özellik 1')
ax1.set_ylabel('Özellik 2')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)
ax1.set_xlim(-7, 7)
ax1.set_ylim(-7, 7)
ax2 = axes[1]
ax2.contourf(xx, yy, Z, levels=25, cmap='RdYlGn', alpha=0.35)
mask_TP = (y_pred == 1) & (y_true == 1)
mask_TN = (y_pred == -1) & (y_true == -1)
mask_FP = (y_pred == 1) & (y_true == -1)
mask_FN = (y_pred == -1) & (y_true == 1)
ax2.scatter(X_test[mask_TP, 0], X_test[mask_TP, 1], c='steelblue', s=30, alpha=0.7, label=f' TP Normal ({mask_TP.sum()})')
ax2.scatter(X_test[mask_TN, 0], X_test[mask_TN, 1], c='red', marker='X', s=100, label=f' TN Anomali ({mask_TN.sum()})')
ax2.scatter(X_test[mask_FP, 0], X_test[mask_FP, 1], c='orange', marker='X', s=150, edgecolors='black', label=f' FP Kaçan ({mask_FP.sum()})', zorder=6)
ax2.scatter(X_test[mask_FN, 0], X_test[mask_FN, 1], c='purple', marker='o', s=100, edgecolors='black', label=f' FN Yanlış Alarm ({mask_FN.sum()})', zorder=6)
ax2.set_title(f'Test Sonuçları — Doğruluk: %{accuracy:.1f} | AUC: {auc:.3f}')
ax2.set_xlabel('Özellik 1')
ax2.set_ylabel('Özellik 2')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)
ax2.set_xlim(-7, 7)
ax2.set_ylim(-7, 7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'decision_surface.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' isolation_forest_sonuclar.png kaydedildi')


# 10. Contamination Parametresinin Etkisi

contamination_values = [0.05, 0.1, 0.15, 0.2, 0.3]
results_cont = []
for cont in contamination_values:
    m = IsolationForest(n_estimators=200, contamination=cont, random_state=42)
    m.fit(X_train_scaled)
    preds = m.predict(X_test_scaled)
    tp = ((preds == 1) & (y_true == 1)).sum()
    tn = ((preds == -1) & (y_true == -1)).sum()
    fp = ((preds == 1) & (y_true == -1)).sum()
    fn = ((preds == -1) & (y_true == 1)).sum()
    acc = (preds == y_true).mean() * 100
    rec = tn / (tn + fn) * 100 if tn + fn > 0 else 0
    results_cont.append({'cont': cont, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Acc': acc, 'Recall': rec})
print('┌────────────┬──────┬──────┬──────┬──────┬──────────┬────────────────┐')
print('│ contaminat │  TP  │  TN  │  FP  │  FN  │ Doğruluk │ Anomali Recall │')
print('├────────────┼──────┼──────┼──────┼──────┼──────────┼────────────────┤')
for r in results_cont:
    marker = ' ◀' if abs(r['cont'] - 0.1) < 0.001 else '  '
    print(f"│    {r['cont']:.2f}    │ {r['TP']:4d} │ {r['TN']:4d} │ {r['FP']:4d} │ {r['FN']:4d} │  %{r['Acc']:5.1f}   │    %{r['Recall']:5.1f}       │{marker}")
print('└────────────┴──────┴──────┴──────┴──────┴──────────┴────────────────┘')
print('\n◀ Seçilen değer | TP=Doğru Normal | TN=Doğru Anomali | FP=Kaçan | FN=Yanlış Alarm')


# 11. Isolation Forest vs One-Class SVM Karşılaştırması

from sklearn.svm import OneClassSVM
ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
ocsvm.fit(X_train_scaled)
pred_svm = ocsvm.predict(X_test_scaled)
acc_svm = (pred_svm == y_true).mean() * 100
iforest = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
iforest.fit(X_train_scaled)
pred_if = iforest.predict(X_test_scaled)
acc_if = (pred_if == y_true).mean() * 100
Z_svm = ocsvm.decision_function(grid_scaled).reshape(xx.shape)
Z_if = iforest.score_samples(grid_scaled).reshape(xx.shape)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('One-Class SVM vs Isolation Forest — Karar Sınırı Karşılaştırması', fontsize=13, fontweight='bold')
models_data = [('One-Class SVM', Z_svm, pred_svm, acc_svm, 'PuBuGn'), ('Isolation Forest', Z_if, pred_if, acc_if, 'RdYlGn')]
for ax, (name, Z_val, preds, acc_val, cmap) in zip(axes, models_data):
    ax.contourf(xx, yy, Z_val, levels=20, cmap=cmap, alpha=0.4)
    correct = preds == y_true
    wrong = ~correct
    mask = correct & (y_true == 1)
    ax.scatter(X_test[mask, 0], X_test[mask, 1], c='steelblue', s=25, alpha=0.7, label=f' Normal ({mask.sum()})')
    mask = correct & (y_true == -1)
    ax.scatter(X_test[mask, 0], X_test[mask, 1], c='red', marker='X', s=80, label=f' Anomali ({mask.sum()})')
    ax.scatter(X_test[wrong, 0], X_test[wrong, 1], c='black', marker='*', s=150, label=f' Hata ({wrong.sum()})', zorder=6)
    ax.set_title(f'{name}\nDoğruluk: %{acc_val:.1f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Özellik 1')
    ax.set_ylabel('Özellik 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' model_karsilastirma.png kaydedildi')
print(f'\nOne-Class SVM  Doğruluk: %{acc_svm:.1f}')
print(f'Isolation Forest Doğruluk: %{acc_if:.1f}')


# 12. Özet ve Sonuçlar

print('=' * 55)
print('        ISOLATION FOREST — SONUÇ ÖZETİ')
print('=' * 55)
print(f'  Eğitim seti      : {len(X_train)} normal nokta (2 küme)')
print(f'  Test seti        : {len(X_test)} nokta ({len(X_test_normal)} normal + {len(X_test_anomaly)} anormal)')
print(f'  Ağaç sayısı      : {model.n_estimators}')
print(f'  Contamination    : {model.contamination}')
print(f'  Genel Doğruluk   : %{accuracy:.1f}')
print(f'  ROC-AUC Skoru    : {auc:.4f}')
print(f'  Anomali Recall   : %{recall * 100:.1f}  ({TN}/{len(X_test_anomaly)} anomali tespit edildi)')
print('=' * 55)
print('\n çalışma başarıyla tamamlandı!')

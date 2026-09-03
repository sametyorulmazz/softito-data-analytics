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
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['grid.alpha'] = 0.3
print(' Tüm kütüphaneler başarıyla yüklendi.')


# 2. Copula Teorisi — Kısa Özet

np.random.seed(42)
n_demo = 200
cov_matrix = [[1.0, 0.8], [0.8, 1.0]]
X_demo = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=n_demo)
U1 = stats.rankdata(X_demo[:, 0]) / n_demo
U2 = stats.rankdata(X_demo[:, 1]) / n_demo
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('ECDF Dönüşümü — Copula Teorisinin Temeli', fontsize=13, fontweight='bold')
ax = axes[0]
ax.scatter(X_demo[:, 0], X_demo[:, 1], alpha=0.5, s=20, c='steelblue')
corr = np.corrcoef(X_demo[:, 0], X_demo[:, 1])[0, 1]
ax.set_title(f'Ham Veri\n(Pearson korelasyon: r={corr:.2f})')
ax.set_xlabel('Özellik 1 (Gauss dağılımı)')
ax.set_ylabel('Özellik 2 (Gauss dağılımı)')
ax.grid(True)
ax = axes[1]
ax.scatter(U1, U2, alpha=0.5, s=20, c='teal')
corr_u = np.corrcoef(U1, U2)[0, 1]
ax.set_title(f'ECDF Sonrası — Copula Uzayı\n(Spearman korelasyon ≈ r={corr_u:.2f})')
ax.set_xlabel('U1 = ECDF(Özellik 1) ∈ [0,1]')
ax.set_ylabel('U2 = ECDF(Özellik 2) ∈ [0,1]')
ax.axvline(0.05, color='red', linestyle='--', alpha=0.5, label='Sol kuyruk (p<0.05)')
ax.axvline(0.95, color='orange', linestyle='--', alpha=0.5, label='Sağ kuyruk (p>0.95)')
ax.axhline(0.05, color='red', linestyle='--', alpha=0.5)
ax.axhline(0.95, color='orange', linestyle='--', alpha=0.5)
ax.legend(fontsize=7)
ax.grid(True)
ax = axes[2]
log_score = np.maximum(-np.log(U1 + 1e-10), -np.log(1 - U1 + 1e-10)) + np.maximum(-np.log(U2 + 1e-10), -np.log(1 - U2 + 1e-10))
sc = ax.scatter(X_demo[:, 0], X_demo[:, 1], c=log_score, cmap='YlOrRd', s=30, alpha=0.8)
plt.colorbar(sc, ax=ax, label='Log-Anomali Skoru')
ax.set_title('Ham Veri Üzerinde Anomali Skoru\n(kırmızı = yüksek skor = şüpheli)')
ax.set_xlabel('Özellik 1')
ax.set_ylabel('Özellik 2')
ax.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'copula_theory.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' copula_teori.png kaydedildi')
print(f'\nHam veri korelasyonu   : {corr:.4f}')
print(f'Copula uzayı korelasyonu: {corr_u:.4f}')
print('→ Korelasyon yapısı ECDF dönüşümünden SONRA da korunur (Sklar Teoremi)')


# 3. Veri Üretimi — Gerçekçi Senaryo

np.random.seed(42)
n_train = 400
n_normal = 120
n_anomaly = 30
cov = [[1.0, 0.7, 0.65, 0.3], [0.7, 1.0, 0.5, 0.2], [0.65, 0.5, 1.0, 0.4], [0.3, 0.2, 0.4, 1.0]]
X_train = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=cov, size=n_train)
X_test_normal = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=cov, size=n_normal)
X_anom_extreme = np.random.multivariate_normal(mean=[3.5, 3.5, 3.5, 3.5], cov=np.eye(4) * 0.3, size=10)
X_anom_corr = np.column_stack([np.random.uniform(3, 5, 10), np.random.uniform(-3, -1, 10), np.random.uniform(3, 5, 10), np.random.uniform(-2, 0, 10)])
X_anom_noise = np.random.uniform(low=-5, high=5, size=(10, 4))
X_test_anomaly = np.vstack([X_anom_extreme, X_anom_corr, X_anom_noise])
X_test = np.vstack([X_test_normal, X_test_anomaly])
y_true = np.array([0] * n_normal + [1] * len(X_test_anomaly))
print(f'Eğitim seti    : {X_train.shape}  → {n_train} normal işlem')
print(f'Test seti      : {X_test.shape}  → {n_normal} normal + {len(X_test_anomaly)} anormal')
print(f'Özellik sayısı : 4 (tutar, frekans, risk, coğrafi sapma)')
print(f'Anomali oranı  : %{len(X_test_anomaly) / len(X_test) * 100:.1f}')
print(f'\nKorelasyon matrisi (eğitim):')
corr_matrix = np.corrcoef(X_train.T)
feat_names = ['Tutar', 'Frekans', 'Risk', 'Coğrafi']
for i, row in enumerate(corr_matrix):
    print(f'  {feat_names[i]:8s}: ' + '  '.join((f'{v:+.2f}' for v in row)))


# 4. Korelasyon Yapısını Görselleştir

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Özellik Çiftleri — Normal vs Anormal Dağılım', fontsize=13, fontweight='bold')
import itertools
pairs = list(itertools.combinations(range(4), 2))
for idx, (i, j) in enumerate(pairs):
    ax = axes[idx // 3][idx % 3]
    ax.scatter(X_train[:, i], X_train[:, j], c='steelblue', s=10, alpha=0.3, label='Normal (eğitim)')
    ax.scatter(X_test_normal[:, i], X_test_normal[:, j], c='royalblue', s=20, alpha=0.6, label='Normal (test)')
    ax.scatter(X_anom_extreme[:, i], X_anom_extreme[:, j], c='red', marker='X', s=80, label='Anom: Uç değer')
    ax.scatter(X_anom_corr[:, i], X_anom_corr[:, j], c='orange', marker='^', s=80, label='Anom: Korelasyon bozucu')
    ax.scatter(X_anom_noise[:, i], X_anom_noise[:, j], c='purple', marker='D', s=50, label='Anom: Gürültü')
    r = np.corrcoef(X_train[:, i], X_train[:, j])[0, 1]
    ax.set_title(f'{feat_names[i]} vs {feat_names[j]}  (r={r:.2f})', fontsize=10)
    ax.set_xlabel(feat_names[i], fontsize=9)
    ax.set_ylabel(feat_names[j], fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'copod_scores.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' ozellik_ciftleri.png kaydedildi')


# 5. Özellik Ölçekleme

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print('Ölçekleme sonuçları (eğitim verisi):')
for i, name in enumerate(feat_names):
    mu = X_train_s[:, i].mean()
    std = X_train_s[:, i].std()
    print(f'  {name:8s}: ortalama={mu:+.6f}  std={std:.6f}')
print('\n Tüm özellikler ortalama≈0 ve std≈1')


# 6. COPOD Modeli — Tanım ve Hiperparametreler

copod = COPOD(contamination=0.2, n_jobs=1)
print(' COPOD modeli oluşturuldu:')
print(f'   contamination : {copod.contamination}')
print(f'   n_jobs        : {copod.n_jobs}')
print()
print(' COPOD parametresi çok azdır — ana güçlü yönü bu!')
print('   Isolation Forest: n_estimators, max_samples, max_features, contamination')
print('   One-Class SVM   : kernel, nu, gamma')
print('   COPOD           : sadece contamination')


# 7. Modeli Eğitme

copod.fit(X_train_s)
n_outliers_train = copod.labels_.sum()
print(f' Model eğitildi — {len(X_train_s)} normal nokta')
print(f'\nEğitim istatistikleri:')
print(f'  Karar Eşiği (threshold) : {copod.threshold_:.4f}')
print(f'  Eğitimde outlier sayısı  : {n_outliers_train} / {len(X_train_s)}')
print(f'  Skor aralığı             : [{copod.decision_scores_.min():.3f}, {copod.decision_scores_.max():.3f}]')
print(f'  Skor ortalaması          : {copod.decision_scores_.mean():.3f}')


# 8. Test Verisi — Tahmin ve Anomali Skorları

y_pred = copod.predict(X_test_s)
scores = copod.decision_function(X_test_s)
proba = copod.predict_proba(X_test_s)
TP = ((y_pred == 0) & (y_true == 0)).sum()
TN = ((y_pred == 1) & (y_true == 1)).sum()
FP = ((y_pred == 0) & (y_true == 1)).sum()
FN = ((y_pred == 1) & (y_true == 0)).sum()
accuracy = (y_pred == y_true).mean() * 100
recall = TN / (TN + FP) * 100 if TN + FP > 0 else 0
precision = TN / (TN + FN) * 100 if TN + FN > 0 else 0
auc = roc_auc_score(y_true, scores)
ap = average_precision_score(y_true, scores)
print(' TEST SONUÇLARI')
print('=' * 50)
print(f'Genel Doğruluk        : %{accuracy:.1f}')
print(f'Anomali Recall        : %{recall:.1f}   (kaç anomali yakalandı?)')
print(f'Anomali Precision     : %{precision:.1f}   (anomali tahminlerinin doğruluğu)')
print(f'ROC-AUC               : {auc:.4f}')
print(f'Average Precision     : {ap:.4f}')
print()
print(f'TP={TP} (doğru normal) | TN={TN} (doğru anomali) | FP={FP} (kaçan) | FN={FN} (yanlış alarm)')
print()
print(f'Skor aralığı — Min: {scores.min():.3f} | Maks: {scores.max():.3f} | Eşik: {copod.threshold_:.3f}')


# 9. Anomali Skorlarının Analizi

scores_normal = scores[y_true == 0]
scores_anomaly = scores[y_true == 1]
scores_extreme = scores[n_normal:n_normal + 10]
scores_corr = scores[n_normal + 10:n_normal + 20]
scores_noise = scores[n_normal + 20:n_normal + 30]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('COPOD Anomali Skoru Analizi', fontsize=13, fontweight='bold')
ax = axes[0]
ax.hist(scores_normal, bins=20, alpha=0.6, color='steelblue', density=True, label=f'Normal (n={len(scores_normal)})')
ax.hist(scores_anomaly, bins=20, alpha=0.6, color='red', density=True, label=f'Anormal (n={len(scores_anomaly)})')
ax.axvline(copod.threshold_, color='orange', linestyle='--', linewidth=2, label=f'Eşik={copod.threshold_:.2f}')
ax.set_xlabel('Anomali Skoru')
ax.set_ylabel('Yoğunluk')
ax.set_title('Skor Dağılımı: Normal vs Anormal')
ax.legend(fontsize=9)
ax.grid(True)
ax = axes[1]
vp = ax.violinplot([scores_normal, scores_extreme, scores_corr, scores_noise], positions=[1, 2, 3, 4], showmedians=True, showextrema=True)
colors_vp = ['steelblue', 'red', 'orange', 'purple']
for patch, color in zip(vp['bodies'], colors_vp):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.axhline(copod.threshold_, color='orange', linestyle='--', linewidth=2, label=f'Eşik={copod.threshold_:.2f}')
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Normal', 'Uç\nDeğer', 'Korelasyon\nBozucu', 'Gürültü'], fontsize=9)
ax.set_ylabel('Anomali Skoru')
ax.set_title('Anomali Tipine Göre Skor Dağılımı')
ax.legend(fontsize=9)
ax.grid(True)
ax = axes[2]
proba_normal = proba[y_true == 0, 1]
proba_anomaly = proba[y_true == 1, 1]
ax.scatter(range(len(proba_normal)), proba_normal, c='steelblue', s=20, alpha=0.6, label=f'Normal (n={len(proba_normal)})')
ax.scatter(range(len(proba_normal), len(proba_normal) + len(proba_anomaly)), proba_anomaly, c='red', marker='X', s=60, label=f'Anormal (n={len(proba_anomaly)})')
ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='Eşik=0.5')
ax.set_xlabel('Nokta İndeksi')
ax.set_ylabel('Anormallik Skoru (predict_proba)')
ax.set_title('Her Nokta için Anormallik Skoru')
ax.legend(fontsize=9)
ax.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_contribution.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' copod_skorlar.png kaydedildi')
print(f'\nOrtalama Skor — Normal: {scores_normal.mean():.3f} | Uç: {scores_extreme.mean():.3f} | Korelasyon: {scores_corr.mean():.3f} | Gürültü: {scores_noise.mean():.3f}')


# 10. COPOD Özellik Bazlı Katkı Analizi

def compute_feature_contributions(model, X_scaled):
    """
    COPOD modelinin her özellik için ürettiği katkı skorunu hesaplar.
    Yüksek katkı → o özellik anomaliyi tetikliyor.
    """
    n, d = X_scaled.shape
    contributions = np.zeros((n, d))
    for j in range(d):
        col = X_scaled[:, j]
        ranks = stats.rankdata(col)
        u = ranks / (n + 1)
        skew_left = -np.log(u)
        skew_right = -np.log(1 - u)
        contributions[:, j] = np.maximum(skew_left, skew_right)
    return contributions
contribs = compute_feature_contributions(copod, X_test_s)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('COPOD — Özellik Bazlı Anomali Katkısı', fontsize=13, fontweight='bold')
sort_idx = np.argsort(scores)[::-1]
top_n = 40
ax = axes[0]
im = ax.imshow(contribs[sort_idx[:top_n]].T, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=ax, label='Katkı Skoru')
ax.set_yticks(range(4))
ax.set_yticklabels(feat_names)
ax.set_xlabel('Test Noktaları (anomali skoruna göre sıralı →)')
ax.set_title(f'İlk {top_n} Noktanın Özellik Katkıları')
true_labels_sorted = y_true[sort_idx[:top_n]]
for k, label in enumerate(true_labels_sorted):
    color = 'red' if label == 1 else 'steelblue'
    ax.axvline(k, color=color, alpha=0.15, linewidth=3)
ax = axes[1]
mean_normal = contribs[y_true == 0].mean(axis=0)
mean_anomaly = contribs[y_true == 1].mean(axis=0)
x = np.arange(4)
width = 0.35
ax.bar(x - width / 2, mean_normal, width, label='Normal', color='steelblue', alpha=0.8)
ax.bar(x + width / 2, mean_anomaly, width, label='Anormal', color='red', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(feat_names)
ax.set_ylabel('Ortalama Katkı Skoru')
ax.set_title('Normal vs Anormal — Özellik Başına Ortalama Katkı')
ax.legend()
ax.grid(True, axis='y')
for i in range(4):
    ratio = mean_anomaly[i] / mean_normal[i] if mean_normal[i] > 0 else 0
    ax.text(i + width / 2, mean_anomaly[i] + 0.05, f'×{ratio:.1f}', ha='center', va='bottom', fontsize=9, color='darkred', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_pairs.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' ozellik_katkisi.png kaydedildi')
print('\nAnormal noktalarda normal noktalara kıyasla özellik katkı oranları:')
for i, name in enumerate(feat_names):
    ratio = mean_anomaly[i] / mean_normal[i] if mean_normal[i] > 0 else 0
    print(f'  {name:10s}: ×{ratio:.2f}')


# 11. Üç Model Karşılaştırması

models = {'COPOD': COPOD(contamination=0.2, n_jobs=1), 'Isolation\nForest': IForest(n_estimators=200, contamination=0.2, random_state=42, n_jobs=1), 'One-Class\nSVM': OCSVM(kernel='rbf', nu=0.2)}
results = {}
for name, m in models.items():
    m.fit(X_train_s)
    preds = m.predict(X_test_s)
    sc_m = m.decision_function(X_test_s)
    tp = ((preds == 0) & (y_true == 0)).sum()
    tn = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 0) & (y_true == 1)).sum()
    fn = ((preds == 1) & (y_true == 0)).sum()
    acc = (preds == y_true).mean() * 100
    rec = tn / (tn + fp) * 100 if tn + fp > 0 else 0
    pre = tn / (tn + fn) * 100 if tn + fn > 0 else 0
    auc_val = roc_auc_score(y_true, sc_m)
    results[name] = {'preds': preds, 'scores': sc_m, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Acc': acc, 'Recall': rec, 'Precision': pre, 'AUC': auc_val}
print('┌────────────────────┬──────────┬──────────┬──────────┬──────────┐')
print('│ Model              │ Doğruluk │  Recall  │Precision │ ROC-AUC  │')
print('├────────────────────┼──────────┼──────────┼──────────┼──────────┤')
for name, r in results.items():
    name_clean = name.replace('\n', ' ')
    print(f"│ {name_clean:18s} │  %{r['Acc']:5.1f}  │  %{r['Recall']:5.1f}  │  %{r['Precision']:5.1f}  │  {r['AUC']:.4f}  │")
print('└────────────────────┴──────────┴──────────┴──────────┴──────────┘')


# 12. ROC Eğrisi ve Model Karşılaştırma Grafikleri

from sklearn.metrics import roc_curve
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Üç Model Karşılaştırması — COPOD vs Isolation Forest vs One-Class SVM', fontsize=12, fontweight='bold')
colors_m = {'COPOD': 'royalblue', 'Isolation\nForest': 'forestgreen', 'One-Class\nSVM': 'crimson'}
ax = axes[0]
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_true, r['scores'])
    ax.plot(fpr, tpr, label=f"{name.replace(chr(10), ' ')} (AUC={r['AUC']:.3f})", color=colors_m[name], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Rastgele (AUC=0.5)')
ax.set_xlabel('Yanlış Pozitif Oranı (FPR)')
ax.set_ylabel('Doğru Pozitif Oranı (TPR)')
ax.set_title('ROC Eğrileri')
ax.legend(fontsize=9)
ax.grid(True)
ax = axes[1]
metric_names = ['Acc', 'Recall', 'Precision']
x = np.arange(len(metric_names))
width = 0.25
for i, (name, r) in enumerate(results.items()):
    vals = [r['Acc'], r['Recall'], r['Precision']]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=name.replace('\n', ' '), color=list(colors_m.values())[i], alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f'%{val:.0f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(['Doğruluk', 'Anomali\nRecall', 'Anomali\nPrecision'])
ax.set_ylabel('Yüzde (%)')
ax.set_ylim(0, 115)
ax.set_title('Metrik Karşılaştırması')
ax.legend(fontsize=8)
ax.grid(True, axis='y')
ax = axes[2]
for i, (name, r) in enumerate(results.items()):
    sc = r['scores']
    sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-10)
    sc_n = sc_norm[y_true == 0]
    sc_a = sc_norm[y_true == 1]
    jitter = np.random.normal(0, 0.04, size=len(sc_n))
    ax.scatter(sc_n, np.full_like(sc_n, i * 2 + 0) + jitter, c='steelblue', s=15, alpha=0.5)
    jitter = np.random.normal(0, 0.04, size=len(sc_a))
    ax.scatter(sc_a, np.full_like(sc_a, i * 2 + 0) + jitter, c='red', s=40, marker='X', alpha=0.8)
ax.set_yticks([0, 2, 4])
ax.set_yticklabels([n.replace('\n', ' ') for n in results.keys()])
ax.set_xlabel('Normalize Anomali Skoru [0,1]')
ax.set_title('Normalize Skor Dağılımı\n(mavi=normal, kırmızı=anormal)')
ax.grid(True, axis='x')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(' model_karsilastirma.png kaydedildi')


# 13. Contamination Parametresinin Etkisi

cont_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
res_cont = []
for cont in cont_values:
    m = COPOD(contamination=cont, n_jobs=1)
    m.fit(X_train_s)
    preds = m.predict(X_test_s)
    sc = m.decision_function(X_test_s)
    tn = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 0) & (y_true == 1)).sum()
    fn = ((preds == 1) & (y_true == 0)).sum()
    acc = (preds == y_true).mean() * 100
    rec = tn / (tn + fp) * 100 if tn + fp > 0 else 0
    auc_v = roc_auc_score(y_true, sc)
    res_cont.append({'cont': cont, 'TN': tn, 'FP': fp, 'FN': fn, 'Acc': acc, 'Recall': rec, 'AUC': auc_v, 'Threshold': m.threshold_})
print('┌────────────┬──────┬──────┬──────┬──────────┬──────────┬──────────┬──────────┐')
print('│ contaminat │  TN  │  FP  │  FN  │ Doğruluk │  Recall  │  AUC     │  Eşik    │')
print('├────────────┼──────┼──────┼──────┼──────────┼──────────┼──────────┼──────────┤')
for r in res_cont:
    marker = ' ◀' if abs(r['cont'] - 0.2) < 0.001 else '  '
    print(f"│    {r['cont']:.2f}    │ {r['TN']:4d} │ {r['FP']:4d} │ {r['FN']:4d} │  %{r['Acc']:5.1f}   │  %{r['Recall']:5.1f}   │  {r['AUC']:.4f}  │  {r['Threshold']:.4f}  │{marker}")
print('└────────────┴──────┴──────┴──────┴──────────┴──────────┴──────────┴──────────┘')
print('\n◀ Seçilen değer | TN=Doğru Anomali | FP=Kaçan | FN=Yanlış Alarm')
print('Not: AUC sabit kalır — contamination sadece eşiği değiştirir, skoru değil!')


# 14. Özet ve Sonuçlar

print('=' * 60)
print('         COPOD — SONUÇ ÖZETİ')
print('=' * 60)
print(f'  Eğitim seti    : {n_train} normal işlem (4 özellik)')
print(f'  Test seti      : {len(X_test)} nokta ({n_normal} normal + {len(X_test_anomaly)} anormal)')
print(f'  Contamination  : {copod.contamination}')
print(f'  Karar Eşiği    : {copod.threshold_:.4f}')
print(f'  Genel Doğruluk : %{accuracy:.1f}')
print(f'  ROC-AUC        : {auc:.4f}')
print(f'  Anomali Recall : %{recall:.1f}  ({TN}/{len(X_test_anomaly)} anomali tespit edildi)')
print()
print('  Anomali tipi bazında ortalama skor:')
print(f'    Uç Değerler          : {scores_extreme.mean():.3f}')
print(f'    Korelasyon Bozucular : {scores_corr.mean():.3f}')
print(f'    Gürültü              : {scores_noise.mean():.3f}')
print(f'    Normal (referans)    : {scores_normal.mean():.3f}')
print('=' * 60)
print('\n çalışma başarıyla tamamlandı!')

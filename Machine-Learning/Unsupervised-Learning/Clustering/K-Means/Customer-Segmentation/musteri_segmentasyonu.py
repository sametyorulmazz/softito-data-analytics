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


# Amaç ve kapsam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

df = pd.read_csv('data/customer_data.csv')
features = df.drop(columns='CUST_ID').copy()
log_cols = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']
features[log_cols] = np.log1p(features[log_cols].clip(lower=0))
prep = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', RobustScaler())])
X = prep.fit_transform(features)
print('Boyut:', df.shape)
print('Eksik değer toplamı:', int(features.isna().sum().sum()))


# Küme sayısının seçimi

rows = []
sample_size = min(3000, len(X))
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
    score = silhouette_score(X, km.labels_, sample_size=sample_size, random_state=42)
    rows.append({'k': k, 'silhouette': score, 'inertia': km.inertia_})
selection = pd.DataFrame(rows)
best_k = int(selection.loc[selection['silhouette'].idxmax(), 'k'])
print('Seçilen k:', best_k)
selection.round(3)

model = KMeans(n_clusters=best_k, n_init=30, random_state=42)
labels = model.fit_predict(X)
segmented = df.assign(cluster=labels)
profile_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']
profiles = segmented.groupby('cluster')[profile_cols].median()
profiles.insert(0, 'count', segmented['cluster'].value_counts().sort_index())
profiles.round(2)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(selection['k'], selection['silhouette'], marker='o')
axes[0].set(title='Küme Sayısı Seçimi', xlabel='k', ylabel='Silhouette')
sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=labels, s=8, alpha=0.45, cmap='tab10')
axes[1].set(title=f'PCA Görünümü (k={best_k})', xlabel='PC1', ylabel='PC2')
plt.tight_layout()
save_figure('customer_segments.png')

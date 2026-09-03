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


# İSPARK Otopark Verisiyle Basit Doğrusal Regresyon (Sıfırdan)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
pd.set_option('display.max_columns', None)


# 1. Veriyi yükleme ve keşif

df = pd.read_csv('data/ispark_parking.csv')
print(df.shape)
df.head()

print(df.isnull().sum())

bad_coords = df[(df['LATITUDE'] < 30) | (df['LONGITUDE'] < 20)]
bad_coords[['PARK_NAME', 'COUNTY_NAME', 'LONGITUDE', 'LATITUDE', 'CAPACITY_OF_PARK']]

df = df[(df['LATITUDE'] > 30) & (df['LONGITUDE'] > 20)].copy()
print('Temizlenmiş satır sayısı:', len(df))

df.groupby('PARK_TYPE_DESC')['CAPACITY_OF_PARK'].agg(['count', 'mean', 'median', 'std'])


# 2. Hipotez ve değişken hazırlama

df['IS_LOT'] = df['PARK_TYPE_DESC'].isin(['AÇIK OTOPARK', 'KAPALI OTOPARK']).astype(int)
df[['PARK_TYPE_DESC', 'IS_LOT']].drop_duplicates()


# 3. Regresyonu sıfırdan hesaplama

x = df['IS_LOT'].values.astype(float)
y = df['CAPACITY_OF_PARK'].values.astype(float)
b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
b0 = y.mean() - b1 * x.mean()
print(f'b0 (intercept) = {b0:.2f}')
print(f'b1 (slope)     = {b1:.2f}')
print(f'Denklem: ŷ = {b0:.1f} + {b1:.1f}·x')


# 4. Model ne kadar iyi? (R²)

y_pred = b0 + b1 * x
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f'R² = {r2:.4f}')


# 5. Doğrulama: scipy ile karşılaştırma

sonuc = stats.linregress(x, y)
print(f'slope (b1)   = {sonuc.slope:.4f}')
print(f'intercept(b0)= {sonuc.intercept:.4f}')
print(f'R²           = {sonuc.rvalue ** 2:.4f}')
print(f'p-değeri     = {sonuc.pvalue:.3e}')


# 6. Görselleştirme

np.random.seed(42)
jitter = np.random.uniform(-0.08, 0.08, size=len(x))
resid = y - y_pred
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(x + jitter, y, alpha=0.4, s=20, color='#3b6ea5', label='Gerçek gözlemler')
xs = np.array([0, 1])
ax.plot(xs, b0 + b1 * xs, color='#d64545', linewidth=2.5, label=f'Regresyon doğrusu: ŷ = {b0:.1f} + {b1:.1f}·x')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Yol Üstü / Taksi / Minibüs (0)', 'Otopark Alanı (1)'])
ax.set_ylabel('Kapasite (araç sayısı)')
ax.set_title('Basit Doğrusal Regresyon: Otopark Tipi → Kapasite')
ax.set_ylim(-100, 2000)
ax.legend(fontsize=8)
ax2 = axes[1]
ax2.scatter(y_pred + np.random.uniform(-2, 2, len(x)), resid, alpha=0.4, s=20, color='#3b6ea5')
ax2.axhline(0, color='#d64545', linewidth=2)
ax2.set_xlabel('Tahmin edilen değer (ŷ)')
ax2.set_ylabel('Artık (residual) = y - ŷ')
ax2.set_title('Artık (Residual) Grafiği')
plt.tight_layout()
save_figure('ispark_regression.png')

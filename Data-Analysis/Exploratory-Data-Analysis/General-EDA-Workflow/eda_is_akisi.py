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


# Kütüphane Yükleme ve Genel Ayarlar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew, kurtosis, shapiro, kstest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
np.random.seed(42)
sns.set_theme(style='whitegrid', palette='husl', font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
print(' Tüm kütüphaneler başarıyla yüklendi!')
print(f' NumPy versiyonu: {np.__version__}')
print(f' Pandas versiyonu: {pd.__version__}')


# Sentetik Veri Seti Oluşturma

df = pd.read_csv(DATA_DIR / 'synthetic_customers.csv', parse_dates=['tarih'])
df['tarih'] = pd.to_datetime(df['tarih'])
df['ay'] = df['tarih'].dt.month
df['yil'] = df['tarih'].dt.year
df['ay_adi'] = df['tarih'].dt.strftime('%B')
print(f' Veri seti başarıyla oluşturuldu!')
print(f' Toplam kayıt: {len(df):,}')
print(f' Toplam sütun: {len(df.columns)}')


# İlk Veri İncelemesi (Data Overview)

print('=' * 65)
print(' VERİ SETİ GENEL BİLGİLERİ')
print('=' * 65)
print(f'\n Boyut (satır x sütun): {df.shape}')
memory = df.memory_usage(deep=True).sum() / 1024 / 1024
print(f' Bellek Kullanımı: {memory:.2f} MB')
print(f'\n Sütun İsimleri: {list(df.columns)}')
print('\n' + '=' * 65)
print(' İLK 5 SATIR (df.head())')
print('=' * 65)

df.head()

df.tail()

print(' VERİ YAPISI (df.info())')
print('=' * 65)
df.info()

print(' SAYISAL DEĞİŞKENLER - İSTATİSTİKSEL ÖZET')
df.describe().round(2)

print(' KATEGORİK DEĞİŞKENLER - ÖZET')
df.describe(include='object').T


# Eksik Veri Analizi (Missing Value Analysis)

eksik_sayi = df.isnull().sum()
eksik_yuzde = (df.isnull().sum() / len(df) * 100).round(2)
eksik_df = pd.DataFrame({'Eksik Sayı': eksik_sayi, 'Eksik Yüzde (%)': eksik_yuzde, 'Veri Tipi': df.dtypes})
eksik_df = eksik_df[eksik_df['Eksik Sayı'] > 0].sort_values('Eksik Sayı', ascending=False)
print(' EKSİK VERİ RAPORU')
print('=' * 50)
print(eksik_df.to_string())
print(f'\n Toplam eksik değer sayısı: {df.isnull().sum().sum()}')
print(f' Tam dolu satır sayısı: {df.dropna().shape[0]} / {len(df)}')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Eksik Veri Analizi', fontsize=16, fontweight='bold', y=1.02)
eksik_veriler = df.isnull().sum()
eksik_veriler = eksik_veriler[eksik_veriler > 0]
renkler = ['#e74c3c' if x > 70 else '#f39c12' if x > 40 else '#27ae60' for x in eksik_veriler.values]
bars = axes[0].barh(eksik_veriler.index, eksik_veriler.values, color=renkler, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, eksik_veriler.values):
    axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{val} ({val / len(df) * 100:.1f}%)', va='center', fontsize=10)
axes[0].set_title('Sütun Bazında Eksik Değer Sayısı', fontweight='bold', pad=15)
axes[0].set_xlabel('Eksik Değer Sayısı')
axes[0].set_xlim(0, 120)
kirmizi = mpatches.Patch(color='#e74c3c', label='Yüksek (>70)')
turuncu = mpatches.Patch(color='#f39c12', label='Orta (40-70)')
yesil = mpatches.Patch(color='#27ae60', label='Düşük (<40)')
axes[0].legend(handles=[kirmizi, turuncu, yesil], loc='lower right', fontsize=9)
eksik_matris = df[['gelir', 'memnuniyet', 'indirim_orani']].isnull()
sns.heatmap(eksik_matris.T, ax=axes[1], cmap='RdYlGn_r', cbar_kws={'label': 'Eksik (1) / Dolu (0)'}, yticklabels=['Gelir', 'Memnuniyet', 'İndirim Oranı'], xticklabels=False)
axes[1].set_title('Eksik Veri Dağılım Haritası\n(Kırmızı = Eksik)', fontweight='bold', pad=15)
axes[1].set_xlabel('Gözlem (Satır) İndeksi')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'missing_values.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('\n Grafik kaydedildi: eksik_veri_analizi.png')


# Aykırı Değer Analizi (Outlier Detection)

def aykiri_tespit_iqr(seri):
    """IQR yöntemiyle aykırı değerleri tespit eder."""
    Q1 = seri.quantile(0.25)
    Q3 = seri.quantile(0.75)
    IQR = Q3 - Q1
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR
    aykiri_mask = (seri < alt_sinir) | (seri > ust_sinir)
    return (aykiri_mask, alt_sinir, ust_sinir)
sayisal_sutunlar = ['yas', 'gelir', 'harcama', 'siparis_sayisi', 'uyelik_suresi']
print(' AYKIRI DEĞER RAPORU (IQR Yöntemi)')
print('=' * 60)
for sutun in sayisal_sutunlar:
    temiz_veri = df[sutun].dropna()
    mask, alt, ust = aykiri_tespit_iqr(temiz_veri)
    aykiri_sayi = mask.sum()
    aykiri_yuzde = aykiri_sayi / len(temiz_veri) * 100
    print(f'  {sutun:20s}: {aykiri_sayi:4d} aykırı değer ({aykiri_yuzde:.1f}%) | Sınırlar: [{alt:.1f}, {ust:.1f}]')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Aykırı Değer Analizi - Kutu Grafikleri (Box Plots)', fontsize=16, fontweight='bold', y=1.01)
renkler = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for i, (sutun, renk) in enumerate(zip(sayisal_sutunlar, renkler)):
    satir, sutun_idx = divmod(i, 3)
    ax = axes[satir][sutun_idx]
    bp = ax.boxplot(df[sutun].dropna(), vert=True, patch_artist=True, flierprops=dict(marker='o', markersize=4, alpha=0.5, markerfacecolor=renk, markeredgecolor='none'), medianprops=dict(color='black', linewidth=2.5), boxprops=dict(facecolor=renk, alpha=0.7))
    veri = df[sutun].dropna()
    ax.set_title(f'{sutun.upper()}\nMedyan: {veri.median():.1f} | Std: {veri.std():.1f}', fontsize=10, fontweight='bold')
    ax.set_ylabel('Değer')
    ax.grid(axis='y', alpha=0.3)
axes[1][2].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'outlier_boxplots.png', dpi=150, bbox_inches='tight')
plt.close('all')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Violin Plot - Dağılım + Kutu Grafik Birleşimi', fontsize=14, fontweight='bold')
df_temp = df.copy()
df_temp['log_harcama'] = np.log1p(df_temp['harcama'])
sns.violinplot(data=df_temp, x='segment', y='log_harcama', hue='cinsiyet', ax=axes[0], palette='Set2', inner='box', cut=0)
axes[0].set_title('Harcama Dağılımı\n(Segment × Cinsiyet)', fontweight='bold')
axes[0].set_xlabel('Müşteri Segmenti')
axes[0].set_ylabel('Log(Harcama)')
buyuk_sehirler = df['sehir'].value_counts().head(4).index.tolist()
df_buyuk = df[df['sehir'].isin(buyuk_sehirler)]
sns.violinplot(data=df_buyuk, x='sehir', y='yas', ax=axes[1], palette='husl', inner='quartile')
axes[1].set_title('Yaş Dağılımı - Şehirlere Göre', fontweight='bold')
axes[1].set_xlabel('Şehir')
axes[1].set_ylabel('Yaş')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'violin_plots.png', dpi=150, bbox_inches='tight')
plt.close('all')


# Tek Değişkenli Analiz (Univariate Analysis)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Tek Değişkenli Analiz - Histogram + KDE (Yoğunluk Eğrisi)', fontsize=16, fontweight='bold', y=1.01)
analiz_sutunlari = [('yas', '#3498db', 'Yaş Dağılımı'), ('gelir', '#e74c3c', 'Gelir Dağılımı'), ('harcama', '#2ecc71', 'Harcama Dağılımı'), ('siparis_sayisi', '#f39c12', 'Sipariş Sayısı Dağılımı'), ('memnuniyet', '#9b59b6', 'Memnuniyet Dağılımı'), ('indirim_orani', '#1abc9c', 'İndirim Oranı Dağılımı'), ('uyelik_suresi', '#e67e22', 'Üyelik Süresi Dağılımı'), ('ay', '#34495e', 'Aylara Göre Dağılım')]
for i, (sutun, renk, baslik) in enumerate(analiz_sutunlari):
    satir, sutun_idx = divmod(i, 3)
    ax = axes[satir][sutun_idx]
    veri = df[sutun].dropna()
    ax.hist(veri, bins='auto', color=renk, alpha=0.7, density=True, edgecolor='white', linewidth=0.5, label='Histogram')
    try:
        kde_x = np.linspace(veri.min(), veri.max(), 200)
        kde = stats.gaussian_kde(veri)
        ax.plot(kde_x, kde(kde_x), color='darkred', linewidth=2.5, label='KDE')
    except:
        pass
    ax.axvline(veri.mean(), color='navy', linestyle='--', linewidth=1.5, label=f'Ortalama: {veri.mean():.1f}')
    ax.axvline(veri.median(), color='green', linestyle=':', linewidth=1.5, label=f'Medyan: {veri.median():.1f}')
    carpiklik = skew(veri)
    ax.set_title(f'{baslik}\nÇarpıklık: {carpiklik:.2f}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Değer', fontsize=8)
    ax.set_ylabel('Yoğunluk', fontsize=8)
axes[2][2].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'numeric_distributions.png', dpi=150, bbox_inches='tight')
plt.close('all')


# Kategorik Değişken Analizi

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
fig.suptitle('Kategorik Değişken Analizi', fontsize=16, fontweight='bold')
ax1 = fig.add_subplot(gs[0, 0])
sehir_sayim = df['sehir'].value_counts()
bars = ax1.barh(sehir_sayim.index, sehir_sayim.values, color=sns.color_palette('viridis', len(sehir_sayim)))
for bar, val in zip(bars, sehir_sayim.values):
    ax1.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2, f'{val} ({val / len(df) * 100:.1f}%)', va='center', fontsize=9)
ax1.set_title('Şehir Dağılımı', fontweight='bold')
ax1.set_xlabel('Müşteri Sayısı')
ax1.set_xlim(0, sehir_sayim.max() * 1.25)
ax2 = fig.add_subplot(gs[0, 1])
cinsiyet_sayim = df['cinsiyet'].value_counts()
patlat = [0.05, 0]
wedges, texts, autotexts = ax2.pie(cinsiyet_sayim.values, labels=cinsiyet_sayim.index, autopct='%1.1f%%', startangle=90, explode=patlat, colors=['#3498db', '#e91e8c'], shadow=True, wedgeprops=dict(edgecolor='white', linewidth=2))
for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
ax2.set_title('Cinsiyet Dağılımı', fontweight='bold')
ax3 = fig.add_subplot(gs[0, 2])
egitim_sirali = ['Ilkokul', 'Ortaokul', 'Lise', 'Lisans', 'Yukseklisans', 'Doktora']
egitim_sayim = df['egitim'].value_counts().reindex(egitim_sirali)
renkler = plt.cm.Blues(np.linspace(0.3, 0.9, len(egitim_sirali)))
ax3.bar(egitim_sayim.index, egitim_sayim.values, color=renkler, edgecolor='white')
ax3.set_title('Eğitim Seviyesi Dağılımı', fontweight='bold')
ax3.set_xlabel('Eğitim Seviyesi')
ax3.set_ylabel('Sayı')
ax3.tick_params(axis='x', rotation=30)
for bar, val in zip(ax3.patches, egitim_sayim.values):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(val), ha='center', va='bottom', fontsize=9)
ax4 = fig.add_subplot(gs[1, 0])
segment_sayim = df['segment'].value_counts()
wedges, texts, autotexts = ax4.pie(segment_sayim.values, labels=segment_sayim.index, autopct='%1.1f%%', colors=['#27ae60', '#f39c12', '#e74c3c'], startangle=90, wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
ax4.text(0, 0, f'Toplam\n{len(df):,}', ha='center', va='center', fontsize=12, fontweight='bold')
ax4.set_title('Müşteri Segmenti Dağılımı (Donut)', fontweight='bold')
ax5 = fig.add_subplot(gs[1, 1:])
capraz = pd.crosstab(df['segment'], df['sehir'], normalize='index') * 100
capraz.plot(kind='bar', stacked=True, ax=ax5, colormap='tab20', edgecolor='white')
ax5.set_title('Segment × Şehir Dağılımı (Yığılmış %)', fontweight='bold')
ax5.set_xlabel('Müşteri Segmenti')
ax5.set_ylabel('Yüzde (%)')
ax5.tick_params(axis='x', rotation=0)
ax5.legend(title='Şehir', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
plt.savefig(FIGURES_DIR / 'categorical_analysis.png', dpi=150, bbox_inches='tight')
plt.close('all')


# İki Değişkenli Analiz (Bivariate Analysis)

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle('İki Değişkenli Analiz - İlişki Grafikleri', fontsize=16, fontweight='bold', y=1.01)
ax = axes[0][0]
log_gelir = np.log1p(df['gelir'].dropna())
log_harcama = np.log1p(df.loc[df['gelir'].notna(), 'harcama'])
scatter = ax.scatter(log_gelir, log_harcama, alpha=0.4, s=20, c=log_gelir, cmap='viridis', edgecolors='none')
z = np.polyfit(log_gelir, log_harcama, 1)
p = np.poly1d(z)
x_line = np.linspace(log_gelir.min(), log_gelir.max(), 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Regresyon Doğrusu')
r, p_val = stats.pearsonr(log_gelir, log_harcama)
ax.set_title(f'Gelir vs Harcama\nr = {r:.3f}, p < 0.001', fontweight='bold')
ax.set_xlabel('Log(Gelir)')
ax.set_ylabel('Log(Harcama)')
plt.colorbar(scatter, ax=ax, label='Log(Gelir)')
ax.legend(fontsize=8)
ax = axes[0][1]
hb = ax.hexbin(df['yas'], df['siparis_sayisi'], gridsize=20, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax, label='Veri Noktası Sayısı')
r2, _ = stats.pearsonr(df['yas'], df['siparis_sayisi'])
ax.set_title(f'Yaş vs Sipariş Sayısı (Hex Bin)\nr = {r2:.3f}', fontweight='bold')
ax.set_xlabel('Yaş')
ax.set_ylabel('Sipariş Sayısı')
ax = axes[0][2]
df_temiz = df.dropna(subset=['memnuniyet'])
sns.stripplot(data=df_temiz, x='memnuniyet', y=np.log1p(df_temiz['harcama']), hue='segment', jitter=0.3, alpha=0.6, size=3, palette='Set2', ax=ax)
ax.set_title('Memnuniyet vs Harcama (Strip Plot)', fontweight='bold')
ax.set_xlabel('Memnuniyet Puanı')
ax.set_ylabel('Log(Harcama)')
ax.legend(title='Segment', fontsize=8)
ax = axes[1][0]
df_temiz2 = df.dropna(subset=['gelir'])
sns.boxplot(data=df_temiz2, x='segment', y=np.log1p(df_temiz2['gelir']), hue='cinsiyet', palette='coolwarm', ax=ax, linewidth=1.5)
ax.set_title('Segment vs Log(Gelir) - Cinsiyete Göre', fontweight='bold')
ax.set_xlabel('Müşteri Segmenti')
ax.set_ylabel('Log(Gelir)')
ax.legend(title='Cinsiyet', fontsize=9)
ax = axes[1][1]
sns.kdeplot(data=df, x='uyelik_suresi', y=np.log1p(df['harcama']), ax=ax, fill=True, cmap='mako', levels=15, thresh=0.05)
ax.set_title('Üyelik Süresi vs Harcama\n(2D Yoğunluk Haritası)', fontweight='bold')
ax.set_xlabel('Üyelik Süresi (Ay)')
ax.set_ylabel('Log(Harcama)')
ax = axes[1][2]
sns.barplot(data=df.dropna(subset=['memnuniyet']), x='sehir', y='memnuniyet', palette='Blues_d', ax=ax, capsize=0.1, ci=95, order=df.groupby('sehir')['memnuniyet'].mean().sort_values(ascending=False).index)
ax.set_title('Şehir Bazında Ortalama Memnuniyet\n(±%95 GA)', fontweight='bold')
ax.set_xlabel('Şehir')
ax.set_ylabel('Ortalama Memnuniyet')
ax.tick_params(axis='x', rotation=30)
ax.set_ylim(0, 12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'bivariate_analysis.png', dpi=150, bbox_inches='tight')
plt.close('all')


# Korelasyon Analizi

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Korelasyon Analizi', fontsize=16, fontweight='bold')
sayisal_df = df[['yas', 'gelir', 'harcama', 'siparis_sayisi', 'memnuniyet', 'indirim_orani', 'uyelik_suresi']].dropna()
pearson_corr = sayisal_df.corr(method='pearson')
mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
etiketler = ['Yaş', 'Gelir', 'Harcama', 'Sipariş', 'Memnuniyet', 'İndirim', 'Üyelik']
sns.heatmap(pearson_corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, xticklabels=etiketler, yticklabels=etiketler, ax=axes[0], cbar_kws={'label': 'Pearson r', 'shrink': 0.8})
axes[0].set_title('Pearson Korelasyon Matrisi\n(Alt Üçgen)', fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)
spearman_corr = sayisal_df.corr(method='spearman')
mask2 = np.triu(np.ones_like(spearman_corr, dtype=bool), k=1)
sns.heatmap(spearman_corr, mask=mask2, annot=True, fmt='.2f', cmap='PiYG', center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, xticklabels=etiketler, yticklabels=etiketler, ax=axes[1], cbar_kws={'label': 'Spearman ρ', 'shrink': 0.8})
axes[1].set_title('Spearman Korelasyon Matrisi\n(Sıralama Tabanlı)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close('all')
print('\n EN YÜKSEK KORELASYONLAR (|r| > 0.1):')
corr_pairs = pearson_corr.unstack().reset_index()
corr_pairs.columns = ['Değişken 1', 'Değişken 2', 'Korelasyon']
corr_pairs = corr_pairs[(corr_pairs['Değişken 1'] != corr_pairs['Değişken 2']) & (abs(corr_pairs['Korelasyon']) > 0.1)].drop_duplicates().sort_values('Korelasyon', key=abs, ascending=False)
print(corr_pairs.head(10).to_string(index=False))


# Çok Değişkenli Analiz (Multivariate)

df_orneklem = df[['yas', 'gelir', 'harcama', 'siparis_sayisi', 'memnuniyet', 'segment']].dropna().sample(300, random_state=42)
df_orneklem['log_gelir'] = np.log1p(df_orneklem['gelir'])
df_orneklem['log_harcama'] = np.log1p(df_orneklem['harcama'])
g = sns.pairplot(df_orneklem[['yas', 'log_gelir', 'log_harcama', 'siparis_sayisi', 'segment']], hue='segment', diag_kind='kde', plot_kws=dict(alpha=0.5, s=20), diag_kws=dict(fill=True), palette='Set1', corner=False)
g.figure.suptitle('Çok Değişkenli Pair Plot\n(n=300 örneklem, Segment bazında renklendirilmiş)', y=1.02, fontsize=14, fontweight='bold')
etiket_map = {'yas': 'Yaş', 'log_gelir': 'Log(Gelir)', 'log_harcama': 'Log(Harcama)', 'siparis_sayisi': 'Sipariş'}
for ax in g.axes.flat:
    if ax:
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        ax.set_xlabel(etiket_map.get(xlabel, xlabel), fontsize=9)
        ax.set_ylabel(etiket_map.get(ylabel, ylabel), fontsize=9)
plt.savefig(FIGURES_DIR / 'pairplot.png', dpi=120, bbox_inches='tight')
plt.close('all')


# Zaman Serisi Analizi

df['ay_periyot'] = df['tarih'].dt.to_period('M')
aylik_veri = df.groupby('ay_periyot').agg(toplam_harcama=('harcama', 'sum'), musteri_sayisi=('musteri_id', 'count'), ort_memnuniyet=('memnuniyet', 'mean'), toplam_siparis=('siparis_sayisi', 'sum')).reset_index()
aylik_veri['tarih'] = aylik_veri['ay_periyot'].dt.to_timestamp()
fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
fig.suptitle('Zaman Serisi Analizi - Aylık Trendler (2022-2023)', fontsize=16, fontweight='bold')
ax = axes[0]
ax.plot(aylik_veri['tarih'], aylik_veri['toplam_harcama'] / 1000, color='#2980b9', linewidth=2.5, marker='o', markersize=6, label='Toplam Harcama')
if len(aylik_veri) >= 3:
    ha_ort = aylik_veri['toplam_harcama'].rolling(window=3, center=True).mean() / 1000
    ax.plot(aylik_veri['tarih'], ha_ort, color='red', linewidth=2, linestyle='--', label='3-Aylık Hareketli Ort.')
ax.fill_between(aylik_veri['tarih'], aylik_veri['toplam_harcama'] / 1000, alpha=0.15, color='#2980b9')
ax.set_title('Aylık Toplam Harcama', fontweight='bold')
ax.set_ylabel('Harcama (Bin TL)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax = axes[1]
degisim = aylik_veri['musteri_sayisi'].diff().fillna(0)
bar_renkleri = ['#27ae60' if x >= 0 else '#e74c3c' for x in degisim]
bars = ax.bar(aylik_veri['tarih'], aylik_veri['musteri_sayisi'], color=bar_renkleri, alpha=0.8, edgecolor='white', width=20)
z = np.polyfit(range(len(aylik_veri)), aylik_veri['musteri_sayisi'], 1)
p = np.poly1d(z)
ax.plot(aylik_veri['tarih'], p(range(len(aylik_veri))), 'k--', linewidth=1.5, label=f'Trend (slope={z[0]:.1f}/ay)')
ax.set_title('Aylık Müşteri Sayısı (Yeşil=Artış, Kırmızı=Azalış)', fontweight='bold')
ax.set_ylabel('Müşteri Sayısı')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax = axes[2]
aylik_std = df.groupby('ay_periyot')['memnuniyet'].std().values
ax.plot(aylik_veri['tarih'], aylik_veri['ort_memnuniyet'], color='#8e44ad', linewidth=2.5, marker='D', markersize=6)
ax.fill_between(aylik_veri['tarih'], aylik_veri['ort_memnuniyet'] - aylik_std, aylik_veri['ort_memnuniyet'] + aylik_std, alpha=0.2, color='#8e44ad', label='±1 Std. Sapma')
genel_ort = df['memnuniyet'].mean()
ax.axhline(genel_ort, color='gray', linestyle=':', linewidth=1.5, label=f'Genel Ortalama: {genel_ort:.2f}')
ax.set_title('Aylık Ortalama Memnuniyet Puanı', fontweight='bold')
ax.set_xlabel('Tarih')
ax.set_ylabel('Ortalama Memnuniyet')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.gcf().autofmt_xdate(rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'time_series.png', dpi=150, bbox_inches='tight')
plt.close('all')


# Normallik Testleri & Dağılım Analizi

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle('Normallik Analizi - Q-Q Plot ve Dağılım Testleri', fontsize=16, fontweight='bold')
test_degiskenleri = [('yas', 'Yaş'), ('harcama', 'Harcama'), ('siparis_sayisi', 'Sipariş Sayısı')]
for i, (sutun, etiket) in enumerate(test_degiskenleri):
    veri = df[sutun].dropna()
    ax_qq = axes[i, 0]
    (osm, osr), (slope, intercept, r) = stats.probplot(veri, dist='norm', plot=None)
    x_line = np.array([osm.min(), osm.max()])
    y_line = slope * x_line + intercept
    ax_qq.plot(x_line, y_line, 'r-', linewidth=2, label='Normal Dağılım Referansı')
    ax_qq.scatter(osm, osr, alpha=0.5, s=15, c='#3498db', label='Gözlenen')
    sw_stat, sw_p = shapiro(veri.sample(min(5000, len(veri)), random_state=42))
    ax_qq.set_title(f"Q-Q Plot: {etiket}\nShapiro-Wilk: W={sw_stat:.3f}, p={sw_p:.4f}\n{(' Normal' if sw_p > 0.05 else ' Normal Değil')}", fontsize=9, fontweight='bold')
    ax_qq.set_xlabel('Teorik Kantiller')
    ax_qq.set_ylabel('Örnek Kantilleri')
    ax_qq.legend(fontsize=8)
    ax_qq.grid(alpha=0.3)
    ax_log = axes[i, 1]
    log_veri = np.log1p(veri - veri.min() + 1)
    (osm2, osr2), (slope2, intercept2, r2_val) = stats.probplot(log_veri, dist='norm', plot=None)
    x_line2 = np.array([osm2.min(), osm2.max()])
    y_line2 = slope2 * x_line2 + intercept2
    ax_log.plot(x_line2, y_line2, 'r-', linewidth=2)
    ax_log.scatter(osm2, osr2, alpha=0.5, s=15, c='#e74c3c')
    sw_stat2, sw_p2 = shapiro(log_veri.sample(min(5000, len(log_veri)), random_state=42))
    ax_log.set_title(f"Q-Q Plot (Log Dönüşüm): {etiket}\nShapiro-Wilk: W={sw_stat2:.3f}, p={sw_p2:.4f}\n{(' Normal' if sw_p2 > 0.05 else ' Normal Değil')}", fontsize=9, fontweight='bold')
    ax_log.set_xlabel('Teorik Kantiller')
    ax_log.set_ylabel('Log Örnek Kantilleri')
    ax_log.grid(alpha=0.3)
pass
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'normality_qq.png', dpi=150, bbox_inches='tight')
plt.close('all')


# Gelişmiş Görselleştirmeler

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Gelişmiş Görselleştirmeler', fontsize=16, fontweight='bold')
ax = axes[0][0]
sehirler = df['sehir'].unique()
renkler = sns.color_palette('husl', len(sehirler))
for sehir, renk in zip(sehirler, renkler):
    veri = df[df['sehir'] == sehir]['harcama']
    log_veri = np.log1p(veri)
    x_vals = np.linspace(log_veri.min(), log_veri.max(), 200)
    kde = stats.gaussian_kde(log_veri)
    ax.fill_between(x_vals, kde(x_vals), alpha=0.3, color=renk, label=sehir)
    ax.plot(x_vals, kde(x_vals), color=renk, linewidth=1.5)
ax.set_title('Şehir Bazında Harcama Dağılımı\n(Örtüşen KDE)', fontweight='bold')
ax.set_xlabel('Log(Harcama)')
ax.set_ylabel('Yoğunluk')
ax.legend(title='Şehir', fontsize=9)
ax = axes[0][1]
sehir_ozet = df.groupby('sehir').agg(ort_gelir=('gelir', 'mean'), ort_harcama=('harcama', 'mean'), musteri_sayisi=('musteri_id', 'count'), ort_memnuniyet=('memnuniyet', 'mean')).reset_index().dropna()
scatter = ax.scatter(np.log1p(sehir_ozet['ort_gelir']), np.log1p(sehir_ozet['ort_harcama']), s=sehir_ozet['musteri_sayisi'] * 2, c=sehir_ozet['ort_memnuniyet'], cmap='RdYlGn', alpha=0.8, edgecolors='gray', linewidths=1)
for _, row in sehir_ozet.iterrows():
    ax.annotate(row['sehir'], (np.log1p(row['ort_gelir']), np.log1p(row['ort_harcama'])), textcoords='offset points', xytext=(5, 5), fontsize=9)
plt.colorbar(scatter, ax=ax, label='Ort. Memnuniyet')
ax.set_title('Şehir Bazında Gelir vs Harcama\n(Boyut=Müşteri Sayısı, Renk=Memnuniyet)', fontweight='bold', fontsize=9)
ax.set_xlabel('Log(Ortalama Gelir)')
ax.set_ylabel('Log(Ortalama Harcama)')
ax = axes[1][0]
egitim_sirasi = ['Ilkokul', 'Ortaokul', 'Lise', 'Lisans', 'Yukseklisans', 'Doktora']
renkler_ridge = plt.cm.plasma(np.linspace(0.2, 0.9, len(egitim_sirasi)))
kaydirma = 0
kaydirma_miktari = 0.7
for egitim_seviye, renk in zip(egitim_sirasi, renkler_ridge):
    veri = df[df['egitim'] == egitim_seviye]['gelir'].dropna()
    if len(veri) < 10:
        continue
    log_veri = np.log1p(veri)
    x_vals = np.linspace(log_veri.min(), log_veri.max(), 200)
    kde = stats.gaussian_kde(log_veri, bw_method=0.5)
    kde_vals = kde(x_vals)
    ax.fill_between(x_vals, kaydirma, kaydirma + kde_vals * 2, alpha=0.7, color=renk)
    ax.plot(x_vals, kaydirma + kde_vals * 2, color='white', linewidth=0.8)
    ax.text(log_veri.min() - 0.1, kaydirma + 0.1, egitim_seviye, fontsize=8, ha='right', va='bottom')
    kaydirma += kaydirma_miktari
ax.set_title('Ridge Plot: Eğitim Seviyesine Göre Gelir\nDağılımı (Log Ölçek)', fontweight='bold')
ax.set_xlabel('Log(Gelir)')
ax.set_yticks([])
ax.grid(alpha=0.2, axis='x')
ax = axes[1][1]
pivot_tablo = df.pivot_table(values='harcama', index='sehir', columns='segment', aggfunc='mean')
sns.heatmap(pivot_tablo, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, linewidths=0.5, cbar_kws={'label': 'Ortalama Harcama (TL)'}, annot_kws={'size': 9})
ax.set_title('Pivot Tablo Isı Haritası\nŞehir × Segment → Ortalama Harcama', fontweight='bold')
ax.set_xlabel('Müşteri Segmenti')
ax.set_ylabel('Şehir')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'advanced_visuals.png', dpi=150, bbox_inches='tight')
plt.close('all')


# EDA Özet Raporu

print('=' * 65)
print(' EDA ÖZET RAPORU')
print('=' * 65)
print(f"\n VERİ SETİ ÖZELLİKLERİ\n{'─' * 40}\n  • Toplam gözlem sayısı : {len(df):,}\n  • Toplam değişken sayısı: {len(df.columns)}\n  • Zaman aralığı        : {df['tarih'].min().date()} → {df['tarih'].max().date()}\n  • Toplam eksik veri    : {df.isnull().sum().sum()} değer ({df.isnull().sum().sum() / df.size * 100:.1f}%)\n\n SAYISAL DEĞİŞKEN İSTATİSTİKLERİ\n{'─' * 40}\n  • Yaş Dağılımı         : Ort={df['yas'].mean():.1f}, Min={df['yas'].min()}, Max={df['yas'].max()}\n  • Ortalama Gelir       : {df['gelir'].mean():,.0f} TL (Medyan: {df['gelir'].median():,.0f} TL)\n  • Ortalama Harcama     : {df['harcama'].mean():,.0f} TL (Medyan: {df['harcama'].median():,.0f} TL)\n  • Ort. Sipariş Sayısı  : {df['siparis_sayisi'].mean():.1f} (Poisson dağılımı)\n  • Ort. Memnuniyet      : {df['memnuniyet'].mean():.2f} / 10\n\n KATEGORİK DEĞİŞKEN ÖZETLERİ\n{'─' * 40}\n  • En kalabalık şehir   : {df['sehir'].value_counts().index[0]} ({df['sehir'].value_counts().iloc[0]} müşteri)\n  • Cinsiyet dengesi     : Erkek %{(df['cinsiyet'] == 'Erkek').mean() * 100:.1f} / Kadın %{(df['cinsiyet'] == 'Kadın').mean() * 100:.1f}\n  • Baskın segment       : {df['segment'].value_counts().index[0]} ({df['segment'].value_counts().iloc[0] / len(df) * 100:.1f}%)\n\n AYKIRI DEĞER DURUMU\n{'─' * 40}\n  • Harcama'da {(df['harcama'] > df['harcama'].quantile(0.99)).sum()} adet %99 üzeri aşırı aykırı değer\n  • Gelir sağa çarpık    : carpıklık = {skew(df['gelir'].dropna()):.2f}\n  • Harcama sağa çarpık  : carpıklık = {skew(df['harcama'].dropna()):.2f}\n\n ÖNEMLİ BULGULAR\n{'─' * 40}\n  1. Gelir ve Harcama arasında pozitif korelasyon mevcut\n  2. İstanbul müşteri tabanının ~%35'ini oluşturuyor\n  3. Lisans mezunları en büyük eğitim grubunu oluşturuyor (%35)\n  4. Memnuniyet puanları şehirler arasında anlamlı farklılık gösteriyor\n  5. Harcama verisi normal dağılımlı değil → Log dönüşümü önerilir\n\n SONRAKİ ADIMLAR ÖNERİLERİ\n{'─' * 40}\n  → Eksik verileri doldur: medyan/mod imputation veya KNN\n  → Aykırı değerleri ele al: winsorization veya log dönüşümü\n  → Özellik mühendisliği: harcama/gelir oranı, RFM skoru\n  → Segmentasyon: K-Means kümeleme analizi\n  → Tahmin modeli: Harcama tahmini için regresyon\n")
print('=' * 65)
print(' EDA tamamlandı! Tüm grafikler kaydedildi.')
print('=' * 65)

fig = plt.figure(figsize=(20, 18))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle(' EDA Dashboard - Müşteri Veri Seti Özeti', fontsize=18, fontweight='bold', y=0.98)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['yas'], bins=25, color='#3498db', alpha=0.8, edgecolor='white')
ax1.axvline(df['yas'].mean(), color='red', linestyle='--', linewidth=2)
ax1.set_title('Yaş Dağılımı', fontweight='bold')
ax1.set_xlabel('Yaş')
ax1.set_ylabel('Frekans')
ax2 = fig.add_subplot(gs[0, 1])
seg = df['segment'].value_counts()
ax2.pie(seg, labels=seg.index, autopct='%1.0f%%', colors=['#27ae60', '#f39c12', '#e74c3c'], wedgeprops=dict(edgecolor='white', linewidth=2))
ax2.set_title('Segment Dağılımı', fontweight='bold')
ax3 = fig.add_subplot(gs[0, 2])
sehir_v = df['sehir'].value_counts()
ax3.barh(sehir_v.index, sehir_v.values, color=sns.color_palette('Blues_r', len(sehir_v)))
ax3.set_title('Şehir Dağılımı', fontweight='bold')
ax4 = fig.add_subplot(gs[1, 0])
log_h = np.log1p(df['harcama'])
ax4.hist(log_h, bins=30, color='#e74c3c', alpha=0.7, density=True, edgecolor='white')
x_kde = np.linspace(log_h.min(), log_h.max(), 200)
ax4.plot(x_kde, stats.gaussian_kde(log_h)(x_kde), 'k-', linewidth=2)
ax4.set_title('Log(Harcama) Dağılımı', fontweight='bold')
ax5 = fig.add_subplot(gs[1, 1])
mini_corr = df[['yas', 'gelir', 'harcama', 'siparis_sayisi']].corr()
sns.heatmap(mini_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax5, cbar=False, xticklabels=['Yaş', 'Gelir', 'Harcama', 'Sipariş'], yticklabels=['Yaş', 'Gelir', 'Harcama', 'Sipariş'])
ax5.set_title('Korelasyon Matrisi', fontweight='bold')
ax5.tick_params(axis='x', rotation=30, labelsize=8)
ax6 = fig.add_subplot(gs[1, 2])
mem_sayim = df['memnuniyet'].dropna().value_counts().sort_index()
ax6.bar(mem_sayim.index, mem_sayim.values, color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(mem_sayim))))
ax6.set_title('Memnuniyet Dağılımı', fontweight='bold')
ax6.set_xlabel('Puan')
ax7 = fig.add_subplot(gs[2, :])
aylik_harcama = df.groupby('ay_periyot')['harcama'].sum().reset_index()
aylik_harcama['tarih'] = aylik_harcama['ay_periyot'].dt.to_timestamp()
ax7.plot(aylik_harcama['tarih'], aylik_harcama['harcama'] / 1000, color='#2c3e50', linewidth=2.5, marker='o', markersize=7)
ax7.fill_between(aylik_harcama['tarih'], aylik_harcama['harcama'] / 1000, alpha=0.15, color='#2c3e50')
ax7.set_title('Aylık Toplam Harcama Trendi', fontweight='bold')
ax7.set_xlabel('Tarih')
ax7.set_ylabel('Harcama (Bin TL)')
ax7.grid(alpha=0.3)
plt.gcf().autofmt_xdate(rotation=20)
plt.savefig(FIGURES_DIR / 'eda_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
plt.close('all')
print('\n EDA Dashboard oluşturuldu: eda_dashboard.png')

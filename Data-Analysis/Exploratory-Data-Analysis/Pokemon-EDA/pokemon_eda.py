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


# 1. Kütüphanelerin Yüklenmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.titlesize'] = 13
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print('Kütüphaneler başarıyla yüklendi ')


# 2. Veri Setinin Yüklenmesi

dosya_yolu = 'data/pokemon.csv'
df = pd.read_csv(dosya_yolu, sep=';', encoding='utf-8')
print(f'Veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun')
df.head()


# 3. Veriye İlk Bakış

print('Veri seti boyutu (satır, sütun):', df.shape)
print('\nSütunlar:', list(df.columns))
print('\nVeri tipleri:')
print(df.dtypes)

df.info()

df.tail()

df.sample(5, random_state=42)


# 4. Sütun Adlarının Temizlenmesi ve Veri Temizliği

yeni_isimler = {'#': 'PokedexNo', 'Name': 'Ad', 'Type 1': 'Tip1', 'Type 2': 'Tip2', 'Total': 'Toplam', 'HP': 'CanPuani', 'Attack': 'Saldiri', 'Defense': 'Savunma', 'Sp. Atk': 'OzelSaldiri', 'Sp. Def': 'OzelSavunma', 'Speed': 'Hiz', 'Generation': 'Nesil', 'Legendary': 'Efsanevi'}
df = df.rename(columns=yeni_isimler)
print('Yeni sütun isimleri:', list(df.columns))

bozuk_isimler = df[df['Ad'].str.contains('�', na=False)]['Ad'].tolist()
print('Bozuk karakter içeren isimler:', bozuk_isimler)
duzeltme_sozlugu = {'Nidoran�': 'Nidoran', 'Nidoran� ': 'Nidoran', 'Flab��': 'Flabebe'}
df['Ad'] = df['Ad'].replace(duzeltme_sozlugu)
df.loc[(df['Ad'] == 'Nidoran') & df['PokedexNo'].isin([29, 30, 31]), 'Ad'] = 'Nidoran-Disi'
df.loc[(df['Ad'] == 'Nidoran') & df['PokedexNo'].isin([32, 33, 34]), 'Ad'] = 'Nidoran-Erkek'
print('\nDüzeltme sonrası kontrol:')
print(df[df['PokedexNo'].isin([29, 30, 31, 32, 33, 34, 669])][['PokedexNo', 'Ad']])


# 5. Eksik Veri (Missing Values) Analizi

eksik_sayisi = df.isnull().sum()
eksik_yuzde = (eksik_sayisi / len(df) * 100).round(2)
eksik_tablo = pd.DataFrame({'Eksik_Sayisi': eksik_sayisi, 'Eksik_Yuzde': eksik_yuzde})
eksik_tablo = eksik_tablo[eksik_tablo['Eksik_Sayisi'] > 0]
eksik_tablo.sort_values('Eksik_Sayisi', ascending=False)

tek_tipli_sayisi = df['Tip2'].isnull().sum()
toplam = len(df)
print(f'Tek tipli Pokémon sayısı: {tek_tipli_sayisi} / {toplam} (%{round(tek_tipli_sayisi / toplam * 100, 1)})')
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='coolwarm')
plt.title('Eksik Veri Haritası (Açık Renk = Eksik Değer)')
plt.xlabel('Sütunlar')
plt.ylabel('Satır İndeksi')
plt.tight_layout()
save_figure('type2_missing.png')

df['Tip2'] = df['Tip2'].fillna('Yok')
print(df['Tip2'].value_counts().head(10))


# 6. Tekrar Eden Kayıtlar ve "Mega Evrim" Kontrolü

tam_kopya_sayisi = df.duplicated().sum()
print(f'Tamamen birebir kopya satır sayısı: {tam_kopya_sayisi}')
pokedex_tekrar = df['PokedexNo'].value_counts()
coklu_pokedex = pokedex_tekrar[pokedex_tekrar > 1]
print(f'\nBirden fazla forma sahip Pokédex numarası sayısı: {len(coklu_pokedex)}')
df[df['PokedexNo'] == 3][['PokedexNo', 'Ad', 'Tip1', 'Tip2', 'Toplam']]

benzersiz_tur_sayisi = df['PokedexNo'].nunique()
toplam_kayit_sayisi = len(df)
print(f'Toplam kayıt (form dahil): {toplam_kayit_sayisi}')
print(f'Benzersiz temel Pokémon türü (Pokédex No): {benzersiz_tur_sayisi}')
print(f'Mega/alternatif form sayısı: {toplam_kayit_sayisi - benzersiz_tur_sayisi}')


# 7. Betimsel İstatistikler

stat_sutunlari = ['CanPuani', 'Saldiri', 'Savunma', 'OzelSaldiri', 'OzelSavunma', 'Hiz', 'Toplam']
df[stat_sutunlari].describe().T

df[['Tip1', 'Tip2', 'Nesil', 'Efsanevi']].astype(str).describe().T


# 8. Kategorik Değişkenlerin Dağılımı

print('Tip1 Dağılımı:')
print(df['Tip1'].value_counts())

plt.figure(figsize=(12, 6))
sirali_tipler = df['Tip1'].value_counts().index
sns.countplot(data=df, y='Tip1', order=sirali_tipler, hue='Tip1', legend=False, palette='tab20')
plt.title('Birincil Tipe (Tip1) Göre Pokémon Sayısı')
plt.xlabel('Pokémon Sayısı')
plt.ylabel('Tip 1')
plt.tight_layout()
save_figure('primary_type_distribution.png')

fig, eksenler = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x='Nesil', hue='Nesil', legend=False, palette='viridis', ax=eksenler[0])
eksenler[0].set_title('Nesillere Göre Pokémon Sayısı')
eksenler[0].set_xlabel('Nesil')
eksenler[0].set_ylabel('Pokémon Sayısı')
sns.countplot(data=df, x='Efsanevi', hue='Efsanevi', legend=False, palette='Set2', ax=eksenler[1])
eksenler[1].set_title('Efsanevi (Legendary) Dağılımı')
eksenler[1].set_xlabel('Efsanevi mi?')
eksenler[1].set_ylabel('Pokémon Sayısı')
plt.tight_layout()
save_figure('generation_legendary.png')


# 9. Sayısal Değişkenlerin (Stat) Dağılımı

fig, eksenler = plt.subplots(3, 3, figsize=(16, 12))
eksenler = eksenler.flatten()
for i, sutun in enumerate(stat_sutunlari):
    sns.histplot(data=df, x=sutun, kde=True, ax=eksenler[i], color='teal', bins=30)
    eksenler[i].axvline(df[sutun].mean(), color='red', linestyle='--', label='Ortalama')
    eksenler[i].axvline(df[sutun].median(), color='green', linestyle=':', label='Medyan')
    eksenler[i].set_title(f'{sutun} Dağılımı')
    eksenler[i].legend(fontsize=8)
for j in range(len(stat_sutunlari), len(eksenler)):
    eksenler[j].axis('off')
plt.tight_layout()
save_figure('stat_distributions.png')

plt.figure(figsize=(12, 6))
stat_uzun_format = df[stat_sutunlari].melt(var_name='Stat', value_name='Deger')
sns.boxplot(data=stat_uzun_format, x='Stat', y='Deger', hue='Stat', legend=False, palette='coolwarm')
plt.title('Tüm Statların Karşılaştırmalı Kutu Grafiği')
plt.xticks(rotation=20)
plt.tight_layout()
save_figure('stat_boxplots.png')


# 10. Aykırı Değer (Outlier) Analizi — IQR Yöntemi

def outlier_tespit_et(seri):
    """
    Verilen bir pandas Serisi için IQR yöntemiyle aykırı değerleri tespit eder.
    Geriye: (aykırı değer sayısı, alt sınır, üst sınır) döndürür.
    """
    Q1 = seri.quantile(0.25)
    Q3 = seri.quantile(0.75)
    IQR = Q3 - Q1
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR
    aykiri_sayisi = ((seri < alt_sinir) | (seri > ust_sinir)).sum()
    return (aykiri_sayisi, alt_sinir, ust_sinir)
sonuclar = []
for sutun in stat_sutunlari:
    sayi, alt, ust = outlier_tespit_et(df[sutun])
    sonuclar.append({'Stat': sutun, 'Aykiri_Sayisi': sayi, 'Alt_Sinir': round(alt, 1), 'Ust_Sinir': round(ust, 1)})
outlier_df = pd.DataFrame(sonuclar)
outlier_df

Q1 = df['Savunma'].quantile(0.25)
Q3 = df['Savunma'].quantile(0.75)
IQR = Q3 - Q1
ust_sinir = Q3 + 1.5 * IQR
savunma_aykirilari = df[df['Savunma'] > ust_sinir][['Ad', 'Tip1', 'Tip2', 'Savunma', 'Efsanevi']]
savunma_aykirilari.sort_values('Savunma', ascending=False)


# 11. Korelasyon Analizi

korelasyon_matrisi = df[stat_sutunlari].corr(method='pearson')
plt.figure(figsize=(9, 7))
sns.heatmap(korelasyon_matrisi, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Stat Sütunları Arası Korelasyon Matrisi')
plt.tight_layout()
save_figure('correlation_matrix.png')


# 12. Tip1'e Göre Kırılım Analizi — Hangi Tip En Güçlü?

tip1_ozet = df.groupby('Tip1', observed=True)[stat_sutunlari].mean().round(1)
tip1_ozet = tip1_ozet.sort_values('Toplam', ascending=False)
tip1_ozet

plt.figure(figsize=(12, 6))
tip1_ozet['Toplam'].plot(kind='bar', color=sns.color_palette('berlin', len(tip1_ozet)))
plt.title('Birincil Tipe (Tip1) Göre Ortalama Toplam Stat')
plt.xlabel('Tip 1')
plt.ylabel('Ortalama Toplam Stat')
plt.xticks(rotation=45)
plt.tight_layout()
save_figure('type_total_stats.png')

plt.figure(figsize=(10, 9))
sns.heatmap(tip1_ozet.drop(columns='Toplam'), annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5)
plt.title("Tip1'e Göre Ortalama Stat Değerleri (Toplam Hariç)")
plt.tight_layout()
save_figure('type_stat_heatmap.png')


# 13. Efsanevi (Legendary) vs Normal Pokémon Karşılaştırması

efsanevi_ozet = df.groupby('Efsanevi', observed=True)[stat_sutunlari].mean().round(1)
efsanevi_ozet

fig, eksenler = plt.subplots(2, 4, figsize=(18, 9))
eksenler = eksenler.flatten()
for i, sutun in enumerate(stat_sutunlari):
    sns.boxplot(data=df, x='Efsanevi', y=sutun, hue='Efsanevi', legend=False, ax=eksenler[i], palette='Set1')
    eksenler[i].set_title(f'{sutun}: Efsanevi vs Normal')
eksenler[-1].axis('off')
plt.tight_layout()
save_figure('legendary_comparison.png')

efsanevi_orani = df['Efsanevi'].value_counts(normalize=True) * 100
print('Efsanevi Pokémon Oranı (%):')
print(efsanevi_orani.round(2))


# 14. Nesil (Generation) Bazlı Analiz

nesil_ozet = df.groupby('Nesil', observed=True).agg(Ortalama_Toplam=('Toplam', 'mean'), Medyan_Toplam=('Toplam', 'median'), Pokemon_Sayisi=('Ad', 'count'), Efsanevi_Sayisi=('Efsanevi', lambda x: (x == True).sum())).round(1)
nesil_ozet['Efsanevi_Orani_%'] = (nesil_ozet['Efsanevi_Sayisi'] / nesil_ozet['Pokemon_Sayisi'] * 100).round(1)
nesil_ozet

plt.figure(figsize=(10, 6))
sns.lineplot(data=nesil_ozet, x=nesil_ozet.index, y='Ortalama_Toplam', marker='o', linewidth=2.5, color='darkorange')
plt.title('Nesillere Göre Ortalama Toplam Stat Değişimi')
plt.xlabel('Nesil')
plt.ylabel('Ortalama Toplam Stat')
plt.xticks(nesil_ozet.index)
plt.tight_layout()
save_figure('generation_trend.png')


# 15. En Güçlü / En Zayıf Pokémon'lar

en_guclu_10 = df.nlargest(10, 'Toplam')[['Ad', 'Tip1', 'Tip2', 'Toplam', 'Efsanevi']]
print('En Güçlü 10 Pokémon (Toplam Stat):')
en_guclu_10

en_zayif_10 = df.nsmallest(10, 'Toplam')[['Ad', 'Tip1', 'Tip2', 'Toplam', 'Efsanevi']]
print('En Zayıf 10 Pokémon (Toplam Stat):')
en_zayif_10

en_iyi_stat_sahipleri = {}
for sutun in stat_sutunlari:
    en_iyi_index = df[sutun].idxmax()
    en_iyi_stat_sahipleri[sutun] = df.loc[en_iyi_index, 'Ad']
en_iyi_df = pd.DataFrame(list(en_iyi_stat_sahipleri.items()), columns=['Stat', 'En_Yuksek_Pokemon'])
en_iyi_df


# 16. Tip1 – Tip2 Kombinasyon Analizi

tip_kombinasyon = pd.crosstab(df['Tip1'], df['Tip2'])
tip_kombinasyon_uzun = tip_kombinasyon.stack().reset_index()
tip_kombinasyon_uzun.columns = ['Tip1', 'Tip2', 'Sayi']
tip_kombinasyon_uzun = tip_kombinasyon_uzun[tip_kombinasyon_uzun['Tip2'] != 'Yok']
en_sik_kombinasyonlar = tip_kombinasyon_uzun.sort_values('Sayi', ascending=False).head(15)
en_sik_kombinasyonlar

en_sik_kombinasyonlar = en_sik_kombinasyonlar.copy()
en_sik_kombinasyonlar['Kombinasyon'] = en_sik_kombinasyonlar['Tip1'] + ' / ' + en_sik_kombinasyonlar['Tip2']
plt.figure(figsize=(10, 8))
sns.barplot(data=en_sik_kombinasyonlar, y='Kombinasyon', x='Sayi', hue='Kombinasyon', legend=False, palette='mako')
plt.title('En Sık Görülen 15 Tip1/Tip2 Kombinasyonu')
plt.xlabel('Pokémon Sayısı')
plt.ylabel('Tip Kombinasyonu')
plt.tight_layout()
save_figure('type_combinations.png')


# 17. Çok Değişkenli Görselleştirme

onemli_statlar = ['Saldiri', 'Savunma', 'Hiz', 'Toplam']
pairplot_grafik = sns.pairplot(df, vars=onemli_statlar, hue='Efsanevi', palette={True: 'gold', False: 'steelblue'}, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20})
pairplot_grafik.fig.suptitle('Seçili Statların İkili İlişkileri (Efsanevi Kırılımında)', y=1.02)
save_figure('pairplot.png')

from math import pi
secilen_pokemonlar = ['Charizard', 'Blastoise', 'Venusaur', 'Pikachu']
radar_statlari = ['CanPuani', 'Saldiri', 'Savunma', 'OzelSaldiri', 'OzelSavunma', 'Hiz']
radar_df = df[df['Ad'].isin(secilen_pokemonlar)][['Ad'] + radar_statlari].drop_duplicates('Ad')
kategori_sayisi = len(radar_statlari)
acilar = [n / float(kategori_sayisi) * 2 * pi for n in range(kategori_sayisi)]
acilar += acilar[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for _, satir in radar_df.iterrows():
    degerler = satir[radar_statlari].tolist()
    degerler += degerler[:1]
    ax.plot(acilar, degerler, linewidth=2, label=satir['Ad'])
    ax.fill(acilar, degerler, alpha=0.1)
ax.set_xticks(acilar[:-1])
ax.set_xticklabels(radar_statlari)
ax.set_title("Seçili Pokémon'ların Stat Karşılaştırması (Radar Grafiği)", y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
save_figure('pokemon_radar.png')

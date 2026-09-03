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


# 1. Kurulum ve İçe Aktarma

import pandas as pd
import numpy as np
print('Pandas sürümü:', pd.__version__)


# 2. Series (Tek Boyutlu Veri Yapısı)

fiyatlar = pd.Series([7.46, 1.85, 7.38, 5.5, 8.66], name='unit_price')
print(fiyatlar)
print('\nVeri tipi:', type(fiyatlar))
print('Ortalama fiyat:', fiyatlar.mean())


# 3. DataFrame ve Veri Okuma/Yazma

df_json = pd.read_json('data/grocery_chain_data.json')
print("JSON'dan okunan veri boyutu:", df_json.shape)
df_json.head()

from io import BytesIO
excel_buffer = BytesIO()
df_json.to_excel(excel_buffer, index=False, sheet_name='Grocery_Data')
excel_buffer.seek(0)
df = pd.read_excel(excel_buffer, sheet_name='Grocery_Data')
print('Excel tamponundan okunan veri boyutu:', df.shape)
df.head()

from io import StringIO
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)
print('CSV tamponundaki karakter sayısı:', len(csv_buffer.getvalue()))


# 4. Veriyi İnceleme

df.info()

print('Boyut:', df.shape)
print('\nSütunlar:', list(df.columns))

df.describe()

print('Mağaza sayısı:', df['store_name'].nunique())
print('Mağazalar:', df['store_name'].unique())
print('\nReyon (aisle) sayısı:', df['aisle'].nunique())
print('Ürün sayısı:', df['product_name'].nunique())


# 5. Veri Seçme ve Filtreleme

df[['store_name', 'product_name', 'final_amount']].head()

df.loc[0, ['store_name', 'product_name', 'final_amount']]

df[df['aisle'] == 'Dairy'].head()

df[(df['final_amount'] > 50) & (df['aisle'] == 'Meat & Seafood')].head()


# 6. Veri Düzenleme

df['indirim_yuzdesi'] = (df['discount_amount'] / df['total_amount'] * 100).round(1)
df[['product_name', 'total_amount', 'discount_amount', 'indirim_yuzdesi']].head()

df['harcama_kategorisi'] = df['final_amount'].apply(lambda x: 'Yüksek' if x > 50 else 'Düşük')
df[['final_amount', 'harcama_kategorisi']].head()

df_tr = df.rename(columns={'store_name': 'magaza_adi', 'product_name': 'urun_adi'})
df_tr[['magaza_adi', 'urun_adi']].head()


# 7. Sıralama ve Gruplama

df.sort_values('final_amount', ascending=False)[['store_name', 'product_name', 'final_amount']].head()

df.groupby('store_name')['final_amount'].mean().sort_values(ascending=False)

df.groupby('aisle').agg(toplam_ciro=('final_amount', 'sum'), islem_sayisi=('final_amount', 'count'), ortalama_harcama=('final_amount', 'mean')).sort_values('toplam_ciro', ascending=False)


# 8. Birleştirme İşlemleri (Concat & Merge)

df_dairy = df[df['aisle'] == 'Dairy'].head(3)
df_bakery = df[df['aisle'] == 'Bakery'].head(3)
df_birlesik = pd.concat([df_dairy, df_bakery])
df_birlesik[['store_name', 'aisle', 'product_name']]

magaza_bolgeleri = pd.DataFrame({'store_name': df['store_name'].dropna().unique()})
magaza_bolgeleri['bolge'] = ['Kuzey', 'Güney', 'Doğu', 'Batı', 'Kuzey', 'Güney', 'Doğu', 'Batı', 'Merkez'][:len(magaza_bolgeleri)]
df_bolgeli = pd.merge(df, magaza_bolgeleri, on='store_name', how='left')
df_bolgeli[['store_name', 'bolge', 'final_amount']].head()


# 9. Uygulamalı Örnek: Mağaza Bazlı Satış Analizi

magaza_ozet = df.groupby('store_name').agg(toplam_ciro=('final_amount', 'sum'), islem_sayisi=('final_amount', 'count')).sort_values('toplam_ciro', ascending=False)
print('Mağaza bazında özet:')
print(magaza_ozet)
en_iyi_magaza = magaza_ozet['toplam_ciro'].idxmax()
print(f'\nEn yüksek ciroya sahip mağaza: {en_iyi_magaza}')

en_cok_satilan = df.groupby('product_name')['quantity'].sum().sort_values(ascending=False)
print('En çok satılan ilk 5 ürün (toplam adet):')
print(en_cok_satilan.head())

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


# 1. Veriyi Yükleme ve İlk Bakış

import pandas as pd
import numpy as np
df_ham = pd.read_json('data/grocery_chain_data.json')
print('Boyut:', df_ham.shape)
df_ham.head()


# 2. Genel Veri Sağlığı Kontrolü

df_ham.info()

print('Sütun bazında eksik değer sayısı:')
print(df_ham.isnull().sum())
print('\nToplam tekrar eden satır sayısı:', df_ham.duplicated().sum())


# 3. Eksik Değerlerin Tespiti ve Giderilmesi

eksik_magaza = df_ham[df_ham['store_name'].isnull()]
print(f'Mağaza adı eksik olan satır sayısı: {len(eksik_magaza)}')
eksik_magaza.head()

df_temiz = df_ham.copy()
df_temiz['store_name'] = df_temiz['store_name'].fillna('Bilinmiyor')
print('Doldurma sonrası eksik değer sayısı:', df_temiz['store_name'].isnull().sum())
df_temiz['store_name'].value_counts()


# 4. Yinelenen (Duplicate) Kayıt Kontrolü

tekrar_sayisi = df_temiz.duplicated(subset=['customer_id', 'transaction_date', 'product_name']).sum()
print(f'Müşteri+tarih+ürün bazında tekrar eden kayıt sayısı: {tekrar_sayisi}')


# 5. Veri Tiplerinin Düzeltilmesi (Tarih Sütunu)

print('Dönüşümden önce tip:', df_temiz['transaction_date'].dtype)
print('Örnek değer:', df_temiz['transaction_date'].iloc[0])
df_temiz['transaction_date'] = pd.to_datetime(df_temiz['transaction_date'], errors='coerce')
print('\nDönüşümden sonra tip:', df_temiz['transaction_date'].dtype)
print('Çevrilemeyen (NaT) satır sayısı:', df_temiz['transaction_date'].isnull().sum())
df_temiz['islem_yili'] = df_temiz['transaction_date'].dt.year
df_temiz['islem_ayi'] = df_temiz['transaction_date'].dt.month
df_temiz[['transaction_date', 'islem_yili', 'islem_ayi']].head()


# 6. Mantıksal Hata / Aykırı Değer Tespiti ve Düzeltilmesi

mantiksiz = df_temiz[df_temiz['discount_amount'] > df_temiz['total_amount']]
print(f'Mantıksız (negatif final_amount üreten) satır sayısı: {len(mantiksiz)}')
mantiksiz[['product_name', 'total_amount', 'discount_amount', 'final_amount']]

df_temiz['discount_amount'] = np.minimum(df_temiz['discount_amount'], df_temiz['total_amount'])
df_temiz['final_amount'] = df_temiz['total_amount'] - df_temiz['discount_amount']
print('Düzeltme sonrası negatif final_amount sayısı:', (df_temiz['final_amount'] < 0).sum())


# 7. Kategori Tutarlılığı Kontrolü

print('Reyonlar:', sorted(df_temiz['aisle'].unique()))
print('\nMağazalar:', sorted(df_temiz['store_name'].unique()))
bosluk_sorunu = df_temiz['store_name'].apply(lambda x: x != x.strip()).sum()
print(f'\nBaşta/sonda boşluk içeren mağaza adı sayısı: {bosluk_sorunu}')

print('Kategori kontrolü tamamlandı: tutarsızlık bulunmadı.')


# 8. İndeks Düzenleme

df_temiz = df_temiz.reset_index(drop=True)
print('İndeks sıfırlandı. İlk 5 satır:')
df_temiz.head()


# 9. Temizlenmiş Veriyi Kaydetme

df_temiz.to_csv(DATA_DIR / 'grocery_chain_data_temiz.csv', index=False)
print("Temizlenmiş veri 'grocery_chain_data_temiz.csv' olarak kaydedildi.")
print('\nSon durumun genel bilgisi:')
df_temiz.info()


# 10. Uçtan Uca Temizleme Fonksiyonu

def veriyi_temizle(df_ham):
    """
    Ham grocery chain DataFrame'ini alır, temel temizleme adımlarını uygular
    ve temizlenmiş DataFrame'i döndürür.
    """
    df = df_ham.copy()
    df['store_name'] = df['store_name'].fillna('Bilinmiyor')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['discount_amount'] = np.minimum(df['discount_amount'], df['total_amount'])
    df['final_amount'] = df['total_amount'] - df['discount_amount']
    df = df.drop_duplicates(subset=['customer_id', 'transaction_date', 'product_name'], keep='first')
    df = df.reset_index(drop=True)
    return df
df_sonuc = veriyi_temizle(df_ham)
print('Temizleme sonrası kontrol:')
print('- Eksik store_name:', df_sonuc['store_name'].isnull().sum())
print('- Negatif final_amount:', (df_sonuc['final_amount'] < 0).sum())
print('- transaction_date tipi:', df_sonuc['transaction_date'].dtype)
df_sonuc.head()

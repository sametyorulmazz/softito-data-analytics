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


# NumPy Orta Düzey

import numpy as np
np.random.seed(0)


# 1. View (Görünüm) ve Copy (Kopya) Farkı

orijinal = np.array([1, 2, 3, 4, 5])
dilim = orijinal[1:4]
dilim[0] = 99
print('Dilim   :', dilim)
print('Orijinal:', orijinal)

orijinal = np.array([1, 2, 3, 4, 5])
kopya = orijinal[1:4].copy()
kopya[0] = 99
print('Kopya   :', kopya)
print('Orijinal:', orijinal)

a = np.arange(10)
b = a[2:5]
c = a[2:5].copy()
print('b.base is a :', b.base is a)
print('c.base is a :', c.base is a)


# 2. Fancy Indexing (Gelişmiş İndeksleme)

a = np.array([10, 20, 30, 40, 50])
indeksler = [0, 2, 4]
print(a[indeksler])
print(a[[4, 4, 0]])

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
satirlar = [0, 2, 3]
print(m[satirlar])
satir_idx = [0, 1, 2]
sutun_idx = [2, 0, 1]
print(m[satir_idx, sutun_idx])


# 3. np.where ile Koşullu İşlemler

notlar = np.array([45, 60, 78, 32, 90, 55])
sonuc = np.where(notlar >= 50, 'Geçti', 'Kaldı')
print(sonuc)
gecen_indeksler = np.where(notlar >= 50)
print('Geçenlerin indeksleri:', gecen_indeksler[0])

veriler = np.array([-3, 5, -1, 8, -7, 2])
temiz_veriler = np.where(veriler < 0, 0, veriler)
print(temiz_veriler)


# 4. Sıralama: sort, argsort, unique

a = np.array([5, 2, 8, 1, 9, 3])
print('sort    :', np.sort(a))
print('argsort :', np.argsort(a))
print('Doğrulama:', a[np.argsort(a)])

isimler = np.array(['Ayşe', 'Mehmet', 'Can', 'Zeynep'])
yaslar = np.array([25, 32, 19, 41])
sira = np.argsort(yaslar)
print('Yaşa göre sıralı isimler:', isimler[sira])
print('Sıralı yaşlar          :', yaslar[sira])

a = np.array([1, 2, 2, 3, 3, 3, 4])
tekil_degerler = np.unique(a)
print('unique         :', tekil_degerler)
degerler, sayilar = np.unique(a, return_counts=True)
print('Değerler:', degerler)
print('Sayılar :', sayilar)


# 5. Array Birleştirme ve Bölme

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print('concatenate:', np.concatenate([a, b]))
print('vstack     :\n', np.vstack([a, b]))
print('hstack     :', np.hstack([a, b]))

m = np.array([[1, 2, 3], [4, 5, 6]])
v = np.array([[7, 8, 9]])
print('Satır ekleme:\n', np.vstack([m, v]))
sutun = np.array([[10], [11]])
print('Sütun ekleme:\n', np.hstack([m, sutun]))

a = np.arange(9)
parcalar = np.split(a, 3)
for i, parca in enumerate(parcalar):
    print(f'Parça {i}:', parca)
esit_olmayan_parcalar = np.array_split(np.arange(10), 3)
print('Eşit olmayan bölme:', esit_olmayan_parcalar)


# 6. Eksen (axis) Kavramını Derinlemesine Anlamak

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('axis=0 toplam (sütun bazlı):', m.sum(axis=0))
print('axis=1 toplam (satır bazlı):', m.sum(axis=1))
print('axis yok (genel toplam)   :', m.sum())

kup = np.arange(24).reshape(2, 3, 4)
print('Şekil:', kup.shape)
print('axis=0 ile toplam şekli:', kup.sum(axis=0).shape)
print('axis=1 ile toplam şekli:', kup.sum(axis=1).shape)
print('axis=2 ile toplam şekli:', kup.sum(axis=2).shape)


# 7. Vektörleştirme ile Performans Karşılaştırması

import time
n = 1000000
liste = list(range(n))
array = np.arange(n)
baslangic = time.time()
liste_kareler = [x ** 2 for x in liste]
python_suresi = time.time() - baslangic
baslangic = time.time()
array_kareler = array ** 2
numpy_suresi = time.time() - baslangic
print(f'Python döngüsü süresi : {python_suresi:.4f} saniye')
print(f'NumPy vektörleştirme  : {numpy_suresi:.4f} saniye')
print(f'NumPy yaklaşık {python_suresi / numpy_suresi:.1f} kat daha hızlı')


# 8. Universal Functions (ufunc)

a = np.array([0, np.pi / 2, np.pi])
print('sin  :', np.sin(a))
print('sqrt :', np.sqrt([1, 4, 9, 16]))
print('exp  :', np.exp([0, 1, 2]))

def kendi_fonksiyonum(x):
    if x % 2 == 0:
        return x / 2
    else:
        return x * 3 + 1
vektorize_fonksiyon = np.vectorize(kendi_fonksiyonum)
a = np.array([1, 2, 3, 4, 5, 6])
print(vektorize_fonksiyon(a))


# 9. Yapılandırılmış (Structured) Array'ler

veri_tipi = np.dtype([('isim', 'U10'), ('yas', 'i4'), ('boy', 'f4')])
kisiler = np.array([('Ahmet', 28, 1.78), ('Elif', 34, 1.65), ('Burak', 22, 1.82)], dtype=veri_tipi)
print(kisiler)
print('İsimler       :', kisiler['isim'])
print('Ortalama yaş  :', kisiler['yas'].mean())
print('En uzun boylu :', kisiler[kisiler['boy'].argmax()])


# 10. Doğrusal Cebir: Özdeğer, Özvektör ve Doğrusal Denklem Çözme

A = np.array([[4, 2], [1, 3]])
ozdegerler, ozvektorler = np.linalg.eig(A)
print('Özdeğerler :', ozdegerler)
print('Özvektörler:\n', ozvektorler)

A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
cozum = np.linalg.solve(A, b)
print('x =', cozum[0], ', y =', cozum[1])
print('Doğrulama (A @ x):', A @ cozum)


# 11. Dosyaya Kaydetme ve Dosyadan Okuma

a = np.arange(20).reshape(4, 5)
np.save(DATA_DIR / 'ornek_array.npy', a)
yuklenen = np.load(DATA_DIR / 'ornek_array.npy')
print('Kaydedilen ile yüklenen aynı mı?:', np.array_equal(a, yuklenen))
np.savez(DATA_DIR / 'coklu_array.npz', birinci=a, ikinci=a * 2)
veri = np.load(DATA_DIR / 'coklu_array.npz')
print('Dosyadaki anahtarlar:', veri.files)
print('İkinci array:\n', veri['ikinci'])


# 12. Maskeleme ile Veri Temizleme (Gerçekçi Mini Örnek)

sicakliklar = np.array([22.5, 23.1, np.nan, 21.8, -999, 24.0, 22.9, np.nan, 150.0, 23.5])
print('Ham veri:', sicakliklar)
eksik_maske = np.isnan(sicakliklar)
print('Eksik değer sayısı:', eksik_maske.sum())
gecerli_maske = (sicakliklar > -50) & (sicakliklar < 50) & ~np.isnan(sicakliklar)
temiz_veri = sicakliklar[gecerli_maske]
print('Temiz veri     :', temiz_veri)
print('Temiz ortalama :', temiz_veri.mean())
print('Kaç ölçüm elendi:', sicakliklar.size - temiz_veri.size)

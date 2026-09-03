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


# NumPy İleri Düzey

import numpy as np
np.random.seed(0)


# 1. Bellek Düzeni: strides, C-order ve F-order

a = np.arange(12).reshape(3, 4)
print('Array:\n', a)
print('shape  :', a.shape)
print('strides:', a.strides)

a = np.arange(12).reshape(3, 4)
b = a.T
print('Transpoze şekli  :', b.shape)
print('Transpoze strides:', b.strides)
print('b, a ile bellek paylaşıyor mu?:', b.base is a)

c_array = np.array([[1, 2, 3], [4, 5, 6]], order='C')
f_array = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print('C-order flatten:', c_array.flatten(order='A'))
print('F-order flatten:', f_array.flatten(order='A'))
print('C-order flags C_CONTIGUOUS:', c_array.flags['C_CONTIGUOUS'])
print('F-order flags F_CONTIGUOUS:', f_array.flags['F_CONTIGUOUS'])


# 2. Gelişmiş Broadcasting Kuralları ve `np.newaxis`

a = np.array([1, 2, 3])
b = np.array([[10], [20], [30]])
sonuc = a + b
print('Sonuç şekli:\n', sonuc)

v = np.array([1, 2, 3])
satir_vektor = v[np.newaxis, :]
sutun_vektor = v[:, np.newaxis]
print('Satır vektör şekli:', satir_vektor.shape)
print('Sütun vektör şekli:', sutun_vektor.shape)
a = np.array([1, 5, 9])
b = np.array([2, 6])
fark_matrisi = a[:, np.newaxis] - b[np.newaxis, :]
print('Fark matrisi:\n', fark_matrisi)


# 3. Sliding Window (Kayan Pencere) İşlemleri

from numpy.lib.stride_tricks import sliding_window_view
seri = np.array([1, 3, 5, 7, 9, 11, 13, 15])
pencereler = sliding_window_view(seri, window_shape=3)
print('Pencereler:\n', pencereler)
hareketli_ortalama = pencereler.mean(axis=1)
print('Hareketli ortalama (window=3):', hareketli_ortalama)


# 4. `einsum` ile Genel Tensör İşlemleri

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matris_carpimi = np.einsum('ij,jk->ik', A, B)
print('einsum matris çarpımı:\n', matris_carpimi)
print('Doğrulama (A @ B)   :\n', A @ B)

A = np.array([[1, 2], [3, 4]])
iz = np.einsum('ii->', A)
print('İz (trace):', iz)
print('Doğrulama :', np.trace(A))
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
nokta_carpim = np.einsum('i,i->', a, b)
print('Nokta çarpım:', nokta_carpim)
print('Doğrulama   :', np.dot(a, b))


# 5. Maskeli Array'ler (`np.ma`)

veri = np.array([10, -1, 15, -1, 22, 18, -1])
maskeli_veri = np.ma.masked_equal(veri, -1)
print('Maskeli array:', maskeli_veri)
print('Maskeli ortalama:', maskeli_veri.mean())
print('Normal ortalama (yanlış sonuç):', veri.mean())
print('Maske dizisi:', maskeli_veri.mask)


# 6. Modern Rastgelelik: `Generator` API

rng = np.random.default_rng(seed=42)
print('Rastgele ondalık sayılar :', rng.random(3))
print('Rastgele tam sayılar     :', rng.integers(1, 100, size=5))
print('Normal dağılımdan örnek  :', rng.normal(loc=0, scale=1, size=4))
secenekler = np.array(['yazı', 'tura'])
atislar = rng.choice(secenekler, size=10, p=[0.5, 0.5])
print('10 atış sonucu           :', atislar)
rng2 = np.random.default_rng(seed=42)
print('Aynı seed aynı sonucu üretir:', np.array_equal(rng.random(0), rng2.random(0)))


# 7. Hızlı Fourier Dönüşümü (FFT)

orneklem_hizi = 500
zaman = np.linspace(0, 1, orneklem_hizi, endpoint=False)
sinyal = np.sin(2 * np.pi * 5 * zaman) + 0.5 * np.sin(2 * np.pi * 50 * zaman)
fft_sonucu = np.fft.fft(sinyal)
frekanslar = np.fft.fftfreq(len(sinyal), d=1 / orneklem_hizi)
genlikler = np.abs(fft_sonucu)
pozitif_maske = frekanslar > 0
en_guclu_indeksler = np.argsort(genlikler[pozitif_maske])[-2:]
print('Tespit edilen baskın frekanslar (Hz):', np.sort(frekanslar[pozitif_maske][en_guclu_indeksler]))


# 8. Polinom Uydurma (Curve Fitting)

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1.1, 3.9, 9.2, 15.8, 25.1, 36.2])
katsayilar = np.polyfit(x, y, deg=2)
print('Bulunan katsayılar (a, b, c):', katsayilar)
polinom = np.poly1d(katsayilar)
print('x=6 için tahmin edilen y  :', polinom(6))
print('Gerçek y değerleri  :', y)
print('Modelin tahminleri  :', polinom(x))


# 9. Tarih/Zaman Verisiyle Çalışma (`datetime64`, `timedelta64`)

tarih = np.array('2024-01-15', dtype='datetime64[D]')
print('Tarih:', tarih)
tarih_araligi = np.arange('2024-01-01', '2024-01-10', dtype='datetime64[D]')
print('Tarih aralığı:', tarih_araligi)
fark = np.datetime64('2024-06-01') - np.datetime64('2024-01-01')
print('İki tarih arasındaki gün farkı:', fark)
yeni_tarih = tarih + np.timedelta64(30, 'D')
print('30 gün sonrası:', yeni_tarih)


# 10. Kısmi Sıralama: `argpartition` ile Performans Kazanımı

buyuk_dizi = np.random.randint(0, 1000000, size=1000000)
k = 5
import time
baslangic = time.time()
tam_siralama_sonucu = np.sort(buyuk_dizi)[:k]
sort_suresi = time.time() - baslangic
baslangic = time.time()
kismi_indeksler = np.argpartition(buyuk_dizi, k)[:k]
kismi_sonuc = np.sort(buyuk_dizi[kismi_indeksler])
partition_suresi = time.time() - baslangic
print('Tam sıralama sonucu :', tam_siralama_sonucu)
print('argpartition sonucu :', kismi_sonuc)
print(f'sort süresi        : {sort_suresi:.5f} sn')
print(f'argpartition süresi: {partition_suresi:.5f} sn')


# 11. `apply_along_axis` ile Eksen Bazlı Özel Fonksiyonlar

def aralik_genisligi(satir):
    return satir.max() - satir.min()
m = np.array([[1, 5, 3], [10, 2, 8], [4, 4, 4]])
satir_bazli = np.apply_along_axis(aralik_genisligi, axis=1, arr=m)
print('Her satırın aralık genişliği:', satir_bazli)
sutun_bazli = np.apply_along_axis(aralik_genisligi, axis=0, arr=m)
print('Her sütunun aralık genişliği:', sutun_bazli)


# 12. `memmap` ile Bellek Sığmayan Büyük Dosyalarla Çalışma

sekil = (1000, 1000)
mm = np.memmap('buyuk_veri.dat', dtype='float32', mode='w+', shape=sekil)
mm[:] = np.random.rand(*sekil)
mm.flush()
mm_okuma = np.memmap('buyuk_veri.dat', dtype='float32', mode='r', shape=sekil)
print('Şekil          :', mm_okuma.shape)
print('İlk 3x3 alt blok:\n', mm_okuma[:3, :3])
print('Tüm verinin ortalaması:', mm_okuma.mean())
del mm, mm_okuma
import os
os.remove('buyuk_veri.dat')

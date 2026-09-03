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


# 1. NumPy'a Giriş

import numpy as np
print('NumPy sürümü:', np.__version__)


# 2. Array (Dizi) Oluşturma

liste = [1, 2, 3, 4, 5]
array_1d = np.array(liste)
print('Array:', array_1d)
print('Tipi:', type(array_1d))

matris = np.array([[1, 2, 3], [4, 5, 6]])
print('2D Array:\n', matris)
print('Boyut sayısı (ndim):', matris.ndim)


# 3. Array Özellikleri

a = np.array([[1, 2, 3], [4, 5, 6]])
print('shape :', a.shape)
print('ndim  :', a.ndim)
print('size  :', a.size)
print('dtype :', a.dtype)
print('itemsize:', a.itemsize, 'byte')

b = np.array([1, 2, 3], dtype=np.float64)
print(b)
print(b.dtype)


# 4. Hazır Array Oluşturma Fonksiyonları

sifirlar = np.zeros((2, 3))
print('zeros:\n', sifirlar)
birler = np.ones((3, 2))
print('ones:\n', birler)
birim = np.eye(3)
print('eye:\n', birim)

aralik = np.arange(0, 10, 2)
print('arange:', aralik)
esit_araliklar = np.linspace(0, 1, 5)
print('linspace:', esit_araliklar)


# 5. İndeksleme ve Dilimleme (Slicing)

a = np.array([10, 20, 30, 40, 50])
print(a[0])
print(a[-1])
print(a[1:4])
print(a[:3])
print(a[::2])

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(m[1, 2])
print(m[0])
print(m[:, 1])
print(m[0:2, 0:2])

a = np.array([1, 2, 3, 4, 5, 6])
kosul = a > 3
print('Koşul dizisi:', kosul)
print('Filtrelenmiş:', a[kosul])
print(a[a % 2 == 0])


# 6. Array Şekillendirme (Reshape)

a = np.arange(12)
print('Orijinal:', a)
b = a.reshape(3, 4)
print('3x4 şekline dönüştürülmüş:\n', b)
c = a.reshape(2, 2, 3)
print('2x2x3 şekline dönüştürülmüş:\n', c)
d = b.flatten()
print('Düzleştirilmiş:', d)


# 7. Matematiksel İşlemler ve Broadcasting

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print('Toplama   :', a + b)
print('Çıkarma   :', b - a)
print('Çarpma    :', a * b)
print('Bölme     :', b / a)
print('Üs alma   :', a ** 2)

matris = np.array([[1, 2, 3], [4, 5, 6]])
vektor = np.array([10, 20, 30])
sonuc = matris + vektor
print(sonuc)

a = np.array([1, 2, 3, 4])
print(a + 5)
print(a * 2)


# 8. Toplulaştırma (Aggregate) Fonksiyonları

a = np.array([[1, 2, 3], [4, 5, 6]])
print('Toplam       :', a.sum())
print('Ortalama     :', a.mean())
print('Standart sapma:', a.std())
print('Minimum      :', a.min())
print('Maksimum     :', a.max())
print('Sütun toplamları (axis=0):', a.sum(axis=0))
print('Satır toplamları (axis=1):', a.sum(axis=1))


# 9. Rastgele Sayı Üretimi (random modülü)

np.random.seed(42)
rastgele_array = np.random.rand(3, 3)
print('rand:\n', rastgele_array)
tam_sayilar = np.random.randint(1, 100, size=5)
print('randint:', tam_sayilar)
normal_dagilim = np.random.randn(4)
print('randn:', normal_dagilim)


# 10. Doğrusal Cebir (Linear Algebra) Temelleri

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
carpim = A @ B
print('Matris çarpımı:\n', carpim)
determinant = np.linalg.det(A)
print('Determinant:', determinant)
ters_matris = np.linalg.inv(A)
print('Ters matris:\n', ters_matris)
transpoze = A.T
print('Transpoze:\n', transpoze)

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


# 1. Gerekli Kütüphaneleri İçe Aktarma

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')
print('Kütüphaneler başarıyla yüklendi!')


# 3. RGB ve BGR Formatları Arasında Dönüşüm

bgr_goruntu = np.zeros((400, 400, 3), dtype=np.uint8)
bgr_goruntu[0:133, :] = [255, 0, 0]
bgr_goruntu[133:266, :] = [0, 255, 0]
bgr_goruntu[266:400, :] = [0, 0, 255]
rgb_goruntu = cv2.cvtColor(bgr_goruntu, cv2.COLOR_BGR2RGB)
print(f'BGR formatında ilk piksel: {bgr_goruntu[10, 10]}')
print(f'RGB formatında aynı piksel: {rgb_goruntu[10, 10]}')
print('(Sıra ters çevrildi!)')
fig, akslar = plt.subplots(1, 2, figsize=(12, 4))
akslar[0].imshow(bgr_goruntu)
akslar[0].set_title('BGR Formatı (Yanlış Renkler)')
akslar[0].axis('off')
akslar[1].imshow(rgb_goruntu)
akslar[1].set_title('RGB Formatı (Doğru Renkler)')
akslar[1].axis('off')
plt.tight_layout()
save_figure('bgr_rgb_comparison.png')


# 4. HSV Renk Uzayı

hsv_goruntu = cv2.cvtColor(bgr_goruntu, cv2.COLOR_BGR2HSV)
print(f'BGR formatında mavi piksel: {bgr_goruntu[10, 10]}')
print(f'HSV formatında aynı piksel: {hsv_goruntu[10, 10]}')
print(f'  H (Ton): {hsv_goruntu[10, 10][0]} (0-180 aralığında)')
print(f'  S (Doygunluk): {hsv_goruntu[10, 10][1]} (0-255 aralığında)')
print(f'  V (Değer): {hsv_goruntu[10, 10][2]} (0-255 aralığında)')
print(f'\nBGR formatında yeşil piksel: {bgr_goruntu[150, 10]}')
print(f'HSV formatında aynı piksel: {hsv_goruntu[150, 10]}')
print(f'\nBGR formatında kırmızı piksel: {bgr_goruntu[300, 10]}')
print(f'HSV formatında aynı piksel: {hsv_goruntu[300, 10]}')

h, s, v = cv2.split(hsv_goruntu)
print(f'H kanalı şekli: {h.shape}')
print(f'S kanalı şekli: {s.shape}')
print(f'V kanalı şekli: {v.shape}')
fig, akslar = plt.subplots(1, 3, figsize=(15, 4))
akslar[0].imshow(h, cmap='hsv')
akslar[0].set_title('H Kanalı (Ton)\n0-180: Rengi gösterir')
akslar[0].axis('off')
akslar[1].imshow(s, cmap='gray')
akslar[1].set_title('S Kanalı (Doygunluk)\n0-255: Rengin saflığı')
akslar[1].axis('off')
akslar[2].imshow(v, cmap='gray')
akslar[2].set_title('V Kanalı (Değer)\n0-255: Parlaklık')
akslar[2].axis('off')
plt.tight_layout()
save_figure('hsv_channels.png')

hsv_renkler = np.zeros((100, 180, 3), dtype=np.uint8)
for hue in range(180):
    hsv_renkler[:, hue] = [hue, 255, 255]
rgb_renkler = cv2.cvtColor(hsv_renkler, cv2.COLOR_HSV2RGB)
plt.figure(figsize=(15, 2))
plt.imshow(rgb_renkler)
plt.title('HSV Hue Renk Tablosu (0-180)')
plt.xlabel('Hue Değeri')
plt.xticks([0, 30, 60, 90, 120, 150, 180])
plt.yticks([])
plt.tight_layout()
save_figure('hue_scale.png')
print("HSV'de Temel Renklerin Hue Değerleri (0-180):")
print('  Kırmızı: 0 ve 180')
print('  Sarı: ~30')
print('  Yeşil: ~60')
print('  Cam (Cyan): ~90')
print('  Mavi: ~120')
print('  Mor (Magenta): ~150')


# 5. LAB Renk Uzayı

lab_goruntu = cv2.cvtColor(bgr_goruntu, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_goruntu)
print(f'BGR formatında mavi piksel: {bgr_goruntu[10, 10]}')
print(f'LAB formatında aynı piksel: {lab_goruntu[10, 10]}')
print(f'  L (Parlaklık): {lab_goruntu[10, 10][0]}')
print(f'  a (Yeşil-Kırmızı): {lab_goruntu[10, 10][1]}')
print(f'  b (Mavi-Sarı): {lab_goruntu[10, 10][2]}')
fig, akslar = plt.subplots(1, 3, figsize=(15, 4))
akslar[0].imshow(l, cmap='gray')
akslar[0].set_title('L Kanalı (Parlaklık)')
akslar[0].axis('off')
akslar[1].imshow(a, cmap='RdYlGn_r')
akslar[1].set_title('a Kanalı (Yeşil-Kırmızı)')
akslar[1].axis('off')
akslar[2].imshow(b, cmap='RdYlBu_r')
akslar[2].set_title('b Kanalı (Mavi-Sarı)')
akslar[2].axis('off')
plt.tight_layout()
save_figure('lab_channels.png')


# 6. YCrCb Renk Uzayı (Video ve Sıkıştırma için)

ycrcb_goruntu = cv2.cvtColor(bgr_goruntu, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(ycrcb_goruntu)
fig, akslar = plt.subplots(1, 3, figsize=(15, 4))
akslar[0].imshow(y, cmap='gray')
akslar[0].set_title('Y Kanalı (Lüminans/Parlaklık)')
akslar[0].axis('off')
akslar[1].imshow(cr, cmap='gray')
akslar[1].set_title('Cr Kanalı (Kırmızı Fark)')
akslar[1].axis('off')
akslar[2].imshow(cb, cmap='gray')
akslar[2].set_title('Cb Kanalı (Mavi Fark)')
akslar[2].axis('off')
plt.tight_layout()
save_figure('ycrcb_channels.png')


# 7. Renkli Nesne Oluşturma ve Tespiti

test_goruntu = np.ones((400, 600, 3), dtype=np.uint8) * 200
cv2.rectangle(test_goruntu, (50, 50), (150, 150), (0, 0, 255), -1)
cv2.circle(test_goruntu, (350, 100), 50, (0, 255, 0), -1)
cv2.ellipse(test_goruntu, (500, 100), (60, 40), 0, 0, 360, (255, 0, 0), -1)
cv2.rectangle(test_goruntu, (200, 250), (300, 350), (0, 255, 255), -1)
cv2.rectangle(test_goruntu, (350, 250), (450, 350), (255, 0, 255), -1)
rgb_test = cv2.cvtColor(test_goruntu, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.imshow(rgb_test)
plt.title('Test Görüntüsü (Renkli Nesneler)')
plt.axis('off')
plt.tight_layout()
save_figure('synthetic_colored_objects.png')
print('Renkli nesneler oluşturuldu!')


# 8. HSV Kullanarak Kırmızı Nesne Tespiti

hsv_test = cv2.cvtColor(test_goruntu, cv2.COLOR_BGR2HSV)
alt_kirmizi_1 = np.array([0, 50, 50])
ust_kirmizi_1 = np.array([10, 255, 255])
alt_kirmizi_2 = np.array([170, 50, 50])
ust_kirmizi_2 = np.array([180, 255, 255])
maske_kirmizi_1 = cv2.inRange(hsv_test, alt_kirmizi_1, ust_kirmizi_1)
maske_kirmizi_2 = cv2.inRange(hsv_test, alt_kirmizi_2, ust_kirmizi_2)
maske_kirmizi = cv2.bitwise_or(maske_kirmizi_1, maske_kirmizi_2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
maske_kirmizi = cv2.erode(maske_kirmizi, kernel, iterations=2)
maske_kirmizi = cv2.dilate(maske_kirmizi, kernel, iterations=2)
kirmizi_sonuc = cv2.bitwise_and(test_goruntu, test_goruntu, mask=maske_kirmizi)
fig, akslar = plt.subplots(2, 2, figsize=(14, 10))
akslar[0, 0].imshow(rgb_test)
akslar[0, 0].set_title('Orijinal Görüntü')
akslar[0, 0].axis('off')
akslar[0, 1].imshow(maske_kirmizi, cmap='gray')
akslar[0, 1].set_title('Kırmızı Maske')
akslar[0, 1].axis('off')
akslar[1, 0].imshow(cv2.cvtColor(kirmizi_sonuc, cv2.COLOR_BGR2RGB))
akslar[1, 0].set_title('Kırmızı Nesneler Tespit Edildi')
akslar[1, 0].axis('off')
akslar[1, 1].axis('off')
plt.tight_layout()
save_figure('red_object_mask.png')
beyaz_piksel_sayisi = cv2.countNonZero(maske_kirmizi)
toplam_piksel = maske_kirmizi.shape[0] * maske_kirmizi.shape[1]
yuzde = beyaz_piksel_sayisi / toplam_piksel * 100
print(f'Tespit edilen kırmızı piksel sayısı: {beyaz_piksel_sayisi}')
print(f"Görüntünün {yuzde:.2f}%'si kırmızı")


# 9. Yeşil ve Mavi Nesne Tespiti

alt_yesil = np.array([35, 40, 40])
ust_yesil = np.array([85, 255, 255])
maske_yesil = cv2.inRange(hsv_test, alt_yesil, ust_yesil)
maske_yesil = cv2.erode(maske_yesil, kernel, iterations=2)
maske_yesil = cv2.dilate(maske_yesil, kernel, iterations=2)
alt_mavi = np.array([100, 50, 50])
ust_mavi = np.array([130, 255, 255])
maske_mavi = cv2.inRange(hsv_test, alt_mavi, ust_mavi)
maske_mavi = cv2.erode(maske_mavi, kernel, iterations=2)
maske_mavi = cv2.dilate(maske_mavi, kernel, iterations=2)
yesil_sonuc = cv2.bitwise_and(test_goruntu, test_goruntu, mask=maske_yesil)
mavi_sonuc = cv2.bitwise_and(test_goruntu, test_goruntu, mask=maske_mavi)
fig, akslar = plt.subplots(2, 3, figsize=(15, 10))
akslar[0, 0].imshow(rgb_test)
akslar[0, 0].set_title('Orijinal')
akslar[0, 0].axis('off')
akslar[0, 1].imshow(cv2.cvtColor(kirmizi_sonuc, cv2.COLOR_BGR2RGB))
akslar[0, 1].set_title('Kırmızı Tespiti')
akslar[0, 1].axis('off')
akslar[0, 2].imshow(cv2.cvtColor(yesil_sonuc, cv2.COLOR_BGR2RGB))
akslar[0, 2].set_title('Yeşil Tespiti')
akslar[0, 2].axis('off')
akslar[1, 0].imshow(maske_kirmizi, cmap='gray')
akslar[1, 0].set_title('Kırmızı Maske')
akslar[1, 0].axis('off')
akslar[1, 1].imshow(maske_yesil, cmap='gray')
akslar[1, 1].set_title('Yeşil Maske')
akslar[1, 1].axis('off')
akslar[1, 2].imshow(cv2.cvtColor(mavi_sonuc, cv2.COLOR_BGR2RGB))
akslar[1, 2].set_title('Mavi Tespiti')
akslar[1, 2].axis('off')
plt.tight_layout()
save_figure('multi_color_detection.png')


# 10. Kontur Tespiti ve Nesnelerin Sınırlanması

konturs_kirmizi, _ = cv2.findContours(maske_kirmizi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sonuc_kontur = test_goruntu.copy()
for kontur in konturs_kirmizi:
    alan = cv2.contourArea(kontur)
    if alan > 500:
        x, y, w, h = cv2.boundingRect(kontur)
        cv2.rectangle(sonuc_kontur, (x, y), (x + w, y + h), (0, 255, 0), 2)
        M = cv2.moments(kontur)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(sonuc_kontur, (cx, cy), 5, (255, 0, 0), -1)
        print(f'Nesne bulundu - Konum: ({x}, {y}), Boyut: {w}x{h}, Alan: {alan}')
fig, akslar = plt.subplots(1, 2, figsize=(14, 5))
akslar[0].imshow(rgb_test)
akslar[0].set_title('Orijinal')
akslar[0].axis('off')
akslar[1].imshow(cv2.cvtColor(sonuc_kontur, cv2.COLOR_BGR2RGB))
akslar[1].set_title('Algılanan Kırmızı Nesneler (Bounding Box ve Merkez)')
akslar[1].axis('off')
plt.tight_layout()
save_figure('red_object_bounding_box.png')


# 11. Fit Ellipse (Elipse Uydurma)

konturs_yesil, _ = cv2.findContours(maske_yesil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sonuc_elips = test_goruntu.copy()
for kontur in konturs_yesil:
    alan = cv2.contourArea(kontur)
    if alan > 500:
        elips = cv2.fitEllipse(kontur)
        cv2.ellipse(sonuc_elips, elips, (255, 0, 0), 2)
        print(f'Elips merkezi: {elips[0]}, Eksenler: {elips[1]}, Açı: {elips[2]}')
fig, akslar = plt.subplots(1, 2, figsize=(14, 5))
akslar[0].imshow(rgb_test)
akslar[0].set_title('Orijinal')
akslar[0].axis('off')
akslar[1].imshow(cv2.cvtColor(sonuc_elips, cv2.COLOR_BGR2RGB))
akslar[1].set_title('Yeşil Nesnelere Uydurulan Elipsler')
akslar[1].axis('off')
plt.tight_layout()
save_figure('ellipse_fitting.png')


# 12. Minimum Enclosing Circle (En Küçük Çevresel Daire)

konturs_mavi, _ = cv2.findContours(maske_mavi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sonuc_daire = test_goruntu.copy()
for kontur in konturs_mavi:
    alan = cv2.contourArea(kontur)
    if alan > 500:
        (x, y), yaricap = cv2.minEnclosingCircle(kontur)
        merkez = (int(x), int(y))
        yaricap = int(yaricap)
        cv2.circle(sonuc_daire, merkez, yaricap, (0, 0, 255), 2)
        cv2.circle(sonuc_daire, merkez, 3, (255, 255, 0), -1)
        print(f'Daire merkezi: {merkez}, Yarıçap: {yaricap}')
fig, akslar = plt.subplots(1, 2, figsize=(14, 5))
akslar[0].imshow(rgb_test)
akslar[0].set_title('Orijinal')
akslar[0].axis('off')
akslar[1].imshow(cv2.cvtColor(sonuc_daire, cv2.COLOR_BGR2RGB))
akslar[1].set_title('Mavi Nesneleri Çevresel Daireler')
akslar[1].axis('off')
plt.tight_layout()
save_figure('minimum_enclosing_circle.png')


# 13. Kontur Özellikleri Analizi

konturs_kirmizi, _ = cv2.findContours(maske_kirmizi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print('=' * 60)
print('KIRMIZI NESNELERIN ÖZELLİKLERİ')
print('=' * 60)
for i, kontur in enumerate(konturs_kirmizi):
    alan = cv2.contourArea(kontur)
    if alan > 500:
        cevre = cv2.arcLength(kontur, True)
        x, y, w, h = cv2.boundingRect(kontur)
        hull = cv2.convexHull(kontur)
        hull_alan = cv2.contourArea(hull)
        solidite = alan / hull_alan if hull_alan > 0 else 0
        if w > 0 and h > 0:
            dis_merkezlik = min(w, h) / max(w, h)
        else:
            dis_merkezlik = 0
        sirkülite = 4 * np.pi * alan / cevre ** 2 if cevre > 0 else 0
        print(f'\nNesne #{i}:')
        print(f'  Alan: {alan:.2f} piksel²')
        print(f'  Çevre: {cevre:.2f} piksel')
        print(f'  Bounding Box: {w}×{h}')
        print(f'  Solidite: {solidite:.3f} (0-1 arası)')
        print(f'  Dış Merkezlik: {dis_merkezlik:.3f} (0-1 arası)')
        print(f'  Sirkülite: {sirkülite:.3f} (0-1 arası)')
print('\n' + '=' * 60)


# 14. Basit Nesne Sınıflandırması

def nesne_siniflandir(sirkülite, dis_merkezlik):
    """
    Sirkülite ve dış merkezlik değerlerine göre nesne sınıflandır
    
    Parametreler:
    - sirkülite: 0-1 arası (1'e yakın = yuvarlak, 0'a yakın = uzun)
    - dis_merkezlik: 0-1 arası (1'e yakın = kare, 0'a yakın = ince)
    
    Dönüş:
    - Nesne türü (string)
    """
    if sirkülite > 0.8:
        return 'Yuvarlak/Daire'
    elif dis_merkezlik > 0.7:
        return 'Kare/Dikdörtgen'
    else:
        return 'Ince/Uzun Şekil'
print('NESNE SINIFLANDIRMA SONUÇLARI')
print('=' * 60)
for i, kontur in enumerate(konturs_kirmizi):
    alan = cv2.contourArea(kontur)
    if alan > 500:
        cevre = cv2.arcLength(kontur, True)
        x, y, w, h = cv2.boundingRect(kontur)
        if w > 0 and h > 0:
            dis_merkezlik = min(w, h) / max(w, h)
        else:
            dis_merkezlik = 0
        sirkülite = 4 * np.pi * alan / cevre ** 2 if cevre > 0 else 0
        sinif = nesne_siniflandir(sirkülite, dis_merkezlik)
        print(f'\nNesne #{i}: {sinif}')
        print(f'  Sirkülite: {sirkülite:.3f}')
        print(f'  Dış Merkezlik: {dis_merkezlik:.3f}')
print('\n' + '=' * 60)

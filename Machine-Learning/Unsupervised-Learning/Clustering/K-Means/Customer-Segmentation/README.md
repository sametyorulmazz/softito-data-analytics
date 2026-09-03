# K-Means ile Kredi Kartı Müşteri Segmentasyonu

Kredi kartı kullanım davranışlarını benzer müşteri gruplarına ayıran; veri dönüşümü, küme sayısı seçimi ve segment profillemesini bir araya getiren K-Means çalışması.

## Problem ve yöntem seçimi

Veride hazır bir hedef etiketi bulunmadığı için problem kümeleme olarak ele alındı. K-Means, segmentlerin merkezler etrafında özetlenmesine ve davranış farklarının iş açısından yorumlanmasına olanak verdiği için seçildi.

## Veri seti

Kredi kartı bakiyesi, alışveriş, nakit avans, ödeme ve limit davranışlarını içeren yerel müşteri verisi kullanılır.

| Dosya | Boyut | Özellik sayısı |
|---|---:|---:|
| `data/customer_data.csv` | 8.950 satır | 17 sayısal özellik + müşteri kimliği |

## Analiz akışı

1. Eksik değerleri medyan ile tamamlama
2. Çarpık parasal değişkenlere `log1p` dönüşümü uygulama
3. Aykırı değerlere dayanıklı `RobustScaler` ile ölçekleme
4. `k=2…6` için silhouette ve inertia değerlerini karşılaştırma
5. Seçilen kümeleri PCA düzleminde görselleştirme
6. Her segment için medyan davranış profili çıkarma

## Sonuç

En yüksek silhouette skoru `k=3` için **0,391** olarak bulundu. Bu değer kümelerin tamamen ayrışmadığını, ancak müşteri davranışlarında yorumlanabilir üç ana profil bulunduğunu gösterir. Segmentleri iş kararlarında kullanmadan önce dönemsel kararlılık ve müşteri başına gelir gibi ek değişkenlerle doğrulamak gerekir.

![Küme sayısı seçimi ve PCA müşteri segmentleri](figures/customer_segments.png)

## Çalıştırma

```bash
pip install -r requirements.txt
python musteri_segmentasyonu.py
```

## Teknolojiler

Python, pandas, NumPy, Matplotlib ve scikit-learn.

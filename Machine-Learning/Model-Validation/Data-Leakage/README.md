# Veri Sızıntısı: Yanıltıcı Model Başarısı Deneyi

Özellik seçimi test etiketlerini gördüğünde, tamamen rastgele hedef üzerinde bile yüksek doğruluk elde edilebildiğini gösteren tekrarlı model doğrulama deneyi.

## Deney tasarımı

Her tekrarda 120 gözlem, 2.000 rastgele özellik ve rastgele ikili hedef üretilir. Gerçek bir sinyal bulunmadığı için güvenilir test performansının yaklaşık %50 olması beklenir.

İki akış karşılaştırılır:

- **Sızıntılı akış:** `SelectKBest`, eğitim–test ayrımından önce tüm etiketleri görür.
- **Doğru akış:** özellik seçimi ve lojistik regresyon bir `Pipeline` içinde yalnızca eğitim verisine fit edilir.

## Veri seti

| Dosya | Boyut | Açıklama |
|---|---:|---|
| `data/synthetic_classification_sample.csv` | 120 satır × 201 sütun | Deney yapısını incelemek için azaltılmış örnek |

Kod, 40 farklı sabit tohumla tam boyutlu veriyi yeniden üretir ve iki yöntemin test doğruluklarını karşılaştırır.

## Sonuçlar

| Akış | Ortalama test doğruluğu |
|---|---:|
| Sızıntılı özellik seçimi | %81,2 |
| Pipeline tabanlı doğru akış | %48,3 |

Sızıntılı sonuç gerçekte var olmayan bir tahmin gücü izlenimi oluşturur. Ön işleme, özellik seçimi ve ölçekleme gibi veriden öğrenen tüm adımlar çapraz doğrulama katlarının içinde tutulmalıdır.

![Sızıntılı ve doğru modelleme akışlarının karşılaştırması](figures/data_leakage_comparison.png)

## Çalıştırma

```bash
pip install -r requirements.txt
python veri_sizintisi.py
```

## Teknolojiler

Python, pandas, NumPy, Matplotlib ve scikit-learn.

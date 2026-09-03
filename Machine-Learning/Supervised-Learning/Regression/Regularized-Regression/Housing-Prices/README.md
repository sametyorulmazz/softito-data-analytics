# Konut Fiyat Tahmini - Ridge, Lasso ve Elastic Net

Konutun fiziksel ve kategorik özelliklerinden satış fiyatı tahmin eden, eksik değer işleme ve çapraz doğrulamayı tek Pipeline içinde birleştiren regresyon çalışması.

## Problem ve Yaklaşım

Fiyat dağılımındaki sağa çarpıklığı azaltmak için hedef değişkene log dönüşümü uygulandı. Sayısal özelliklerde medyan tamamlama ve standardizasyon; kategorik özelliklerde en sık değer tamamlama ve one-hot encoding kullanıldı. Tüm dönüşümler veri sızıntısını önlemek için Pipeline içinde yalnızca eğitim katlarında öğrenildi.

## Veri Seti

`data/housing.csv` dosyasında 545 konut ve 13 değişken bulunur. Alan, yatak odası, banyo, kat, otopark, ana yola erişim, klima ve mobilya durumu gibi özellikler kullanılır; hedef `price` sütunudur.

## Model Seçimi

Ridge, Lasso ve Elastic Net hiperparametreleri beş katlı çapraz doğrulamayla karşılaştırıldı. Final seçim test setine bakılarak değil, eğitim verisindeki doğrulama sonuçlarıyla yapıldı.

## Sonuçlar

| Metrik | Ridge |
|---|---:|
| Test MAE | yaklaşık 964.859 |
| Test R² | 0,653 |

Model fiyat varyansının yaklaşık %65,3'ünü açıklamaktadır. Kalan hata; semt, bina yaşı, erişilebilirlik ve piyasa zamanı gibi veri setinde bulunmayan değişkenlerin etkisine işaret eder.

![Konut fiyat modeli sonuçları](figures/housing_model_results.png)

## Çalıştırma

```bash
pip install -r requirements.txt
python housing_regularization.py
```

**Teknolojiler:** Python, pandas, scikit-learn, Matplotlib, Seaborn

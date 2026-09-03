# Pandas ile Uçtan Uca Veri Temizleme

Market zinciri verisindeki eksik, yinelenen, hatalı tipte ve mantıksal olarak tutarsız kayıtları kontrollü adımlarla temizler.

## Veri dosyaları

| Dosya | Rol |
|---|---|
| `data/grocery_chain_data.json` | Ham girdi |
| `data/grocery_chain_data_temiz.csv` | Temizleme akışının çıktısı |

## Temizleme akışı

1. Veri türü ve eksik değer profili çıkarma
2. Yinelenen kayıtları denetleme
3. Tarih sütununu doğru tipe dönüştürme
4. Negatif veya mantık dışı değerleri kontrol etme
5. Kategori yazımlarını standartlaştırma
6. Temiz veriyi CSV olarak kaydetme
7. Aynı adımları tekrar kullanılabilir fonksiyonda birleştirme

## Çalıştırma

```bash
pip install -r requirements.txt
python veri_temizleme.py
```

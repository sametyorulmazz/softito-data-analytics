# Unsupervised Learning

Hedef etiketi olmadan veri içindeki grupları ve sıra dışı gözlemleri keşfeden çalışmalar.

| Alt alan | Amaç | Projeler |
|---|---|---|
| [Clustering](Clustering/) | Benzer gözlemleri segmentlere ayırmak | K-Means müşteri segmentasyonu |
| [Anomaly Detection](Anomaly-Detection/) | Normal örüntüden ayrılan gözlemleri bulmak | COPOD, Isolation Forest, One-Class SVM |

Kümeleme sonucunda silhouette; anomali deneylerinde precision, recall ve ROC-AUC gibi ölçüler yorumlanır. Etiketler varsa model eğitiminde değil, yalnızca son değerlendirmede kullanılır.

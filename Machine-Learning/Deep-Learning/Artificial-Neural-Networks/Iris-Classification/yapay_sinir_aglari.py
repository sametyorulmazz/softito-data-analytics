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


# 2. GEREKLI KÜTÜPHANELERİN YÜKLENMESİ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
print(' Tüm kütüphaneler başarıyla yüklendi!')


# 3. IRIS VERİ SETİNİN YÜKLENMESI

iris = datasets.load_iris()
X = iris.data
y = iris.target
print('Iris Veri Seti Bilgileri:')
print(f' Toplam örnek sayısı: {X.shape[0]}')
print(f' Özellik sayısı: {X.shape[1]}')
print(f' Sınıf sayısı: {len(np.unique(y))}')
print(f' Özellik adları: {iris.feature_names}')
print(f' Sınıf adları: {iris.target_names}')
print(f'\nİlk 3 örnek:')
print(X[:3])
print(f'\nİlk 3 örneğin sınıfları:')
print(y[:3])


# 4. VERİNİN ÖN İŞLENMESİ

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Eğitim seti boyutu: {X_train.shape[0]}')
print(f'Test seti boyutu: {X_test.shape[0]}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(' Veri standardizasyonu tamamlandı!')
print(f'\nStandartlaştırılmış eğitim verisi istatistikleri:')
print(f'Ortalama: {X_train_scaled.mean(axis=0).round(3)}')
print(f'Standart Sapma: {X_train_scaled.std(axis=0).round(3)}')


# 5. YAPAY SİNİR AĞI MODELİ OLUŞTURMA

model = Sequential()
model.add(layers.Dense(units=64, activation='relu', input_dim=4))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(units=16, activation='relu'))
model.add(layers.Dense(units=3, activation='softmax'))
print('Model Mimarisi:')
print('=' * 50)
model.summary()
print('=' * 50)


# 6. MODELİN DERLENMESİ (Compilation)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print(' Model başarıyla derlenmiştir!')
print('\nDerleme Detayları:')
print(f'  Optimizatör: Adam (learning_rate=0.001)')
print(f'  Kayıp Fonksiyonu: Categorical Crossentropy')
print(f'  Metrikler: Accuracy')


# 7. TARGETİ ONE-HOT ENCODING'E ÇEVİRME

y_train_encoded = keras.utils.to_categorical(y_train, num_classes=3)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes=3)
print('One-Hot Encoding Örneği:')
print(f'Orijinal y_train: {y_train[:5]}')
print(f'\nEncoded y_train:')
print(y_train_encoded[:5])


# 8. MODELİN EĞİTİLMESİ

history = model.fit(x=X_train_scaled, y=y_train_encoded, epochs=100, batch_size=8, validation_split=0.2, verbose=1)
print('\n Model eğitimi tamamlandı!')


# 9. EĞİTİM SÜRECİNİN GÖRSELLEŞTİRİLMESİ

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validasyon Loss', linewidth=2)
plt.title('Model Loss (Kayıp) Değişimi', fontsize=12, fontweight='bold')
plt.xlabel('Epoch (Dönem)', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validasyon Accuracy', linewidth=2)
plt.title('Model Doğruluk (Accuracy) Değişimi', fontsize=12, fontweight='bold')
plt.xlabel('Epoch (Dönem)', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_figure('training_curves.png')
print('\n Grafiklerin Yorumu:')
print(' Loss grafiği: Daha düşük değerler, daha iyi model performansı anlamına gelir')
print(' Accuracy grafiği: Daha yüksek değerler, daha iyi sınıflandırma anlamına gelir')
print(' Eğitim ve Validasyon eğrileri yakınsa, model iyi genelleşme yapıyor')


# 10. TEST VERİSİ ÜZERİNDE DEĞERLENDİRME

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
print('Test Seti Performansı:')
print('=' * 50)
print(f'Test Loss (Kayıp): {test_loss:.4f}')
print(f'Test Accuracy (Doğruluk): {test_accuracy:.4f} (%{test_accuracy * 100:.2f})')
print('=' * 50)


# 11. TAHMİNLER VE KARMAŞIKLIK MATRİSİ

y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
print('İlk 10 Tahminin Detayları:')
print('=' * 70)
print('Gerçek Sınıf | Tahmin Edilen Sınıf | Tahmin Güvenliği')
print('-' * 70)
for i in range(10):
    max_prob = np.max(y_pred_proba[i])
    print(f'{iris.target_names[y_test[i]]:12} | {iris.target_names[y_pred[i]]:19} | %{max_prob * 100:.2f}')
print('=' * 70)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Sayı'}, xticklabels=iris.target_names, yticklabels=iris.target_names, linewidths=1, linecolor='black')
plt.title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Sınıf', fontsize=12, fontweight='bold')
plt.xlabel('Tahmin Edilen Sınıf', fontsize=12, fontweight='bold')
plt.tight_layout()
save_figure('confusion_matrix.png')
print('\n Karmaşıklık Matrisinin Yorumu:')
print(' Köşegen (çapraz) değerler: Doğru sınıflandırılmış örnekler')
print(' Köşegen dışı değerler: Yanlış sınıflandırılmış örnekler')


# 12. DETAYLI PERFORMANS RAPORUru

print('\n' + '=' * 70)
print('DETAYLI SINIFLAMA RAPORU')
print('=' * 70)
print(classification_report(y_test, y_pred, target_names=iris.target_names, digits=4))
accuracy = accuracy_score(y_test, y_pred)
print(f'Genel Doğruluk (Overall Accuracy): {accuracy:.4f} (%{accuracy * 100:.2f})')
print('=' * 70)


# 13. ÖZELLİKLERİN AĞIRLIKLARININ GÖRSELLEŞTİRİLMESİ

first_layer_weights = model.layers[0].get_weights()[0]
feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
plt.figure(figsize=(10, 6))
bars = plt.bar(iris.feature_names, feature_importance, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], edgecolor='black', linewidth=1.5)
plt.title('Giriş Özellikleri Önem Sırası', fontsize=14, fontweight='bold')
plt.xlabel('Özellikler', fontsize=12, fontweight='bold')
plt.ylabel('Ortalama Ağırlık (Önem)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
save_figure('input_weight_importance.png')
print('\n Özellik Önem Analizi:')
print('Ağırlığı yüksek özellikler, modelin sınıflandırma kararını daha çok etkiler')
for i, feature in enumerate(iris.feature_names):
    print(f'  {feature:20} : {feature_importance[i]:.4f}')


# 14. YENİ VERİ İLE TAHMİN YAPMA

new_sample = X_test[0:1]
new_sample_scaled = scaler.transform(new_sample)
prediction_proba = model.predict(new_sample_scaled, verbose=0)
predicted_class = np.argmax(prediction_proba, axis=1)[0]
print('Yeni Örneğin Tahmin Sonucu:')
print('=' * 50)
print(f'\nÖzellikleri:')
for i, feature in enumerate(iris.feature_names):
    print(f'  {feature:20} : {new_sample[0][i]:.2f}')
print(f'\nTahmin Edilen Sınıf: {iris.target_names[predicted_class].upper()}')
print(f'Gerçek Sınıf: {iris.target_names[y_test[0]].upper()}')
print(f'\nTüm Sınıflar için Güven Seviyeleri:')
for i, class_name in enumerate(iris.target_names):
    prob = prediction_proba[0][i] * 100
    bar = '█' * int(prob / 5)
    print(f'  {class_name:12} : {prob:5.2f}% {bar}')
print('=' * 50)


# 15. ÖZET VE BULGULAR

print('\n' + ' ' * 20)
print('YAPAY SİNİR AĞI EĞİTİM ÖZETİ')
print(' ' * 20)
print('\n1  VERİ SETI:')
print(f'   • Toplam örnek: 150')
print(f'   • Eğitim seti: {X_train.shape[0]} örnek')
print(f'   • Test seti: {X_test.shape[0]} örnek')
print(f'   • Özellik sayısı: 4')
print(f'   • Sınıf sayısı: 3')
print('\n2  MODEL MİMARİSİ:')
print(f'   • Giriş katmanı: 4 nöron')
print(f'   • 1. Gizli katman: 64 nöron (ReLU aktivasyonu)')
print(f'   • Dropout: %30')
print(f'   • 2. Gizli katman: 32 nöron (ReLU aktivasyonu)')
print(f'   • Dropout: %30')
print(f'   • 3. Gizli katman: 16 nöron (ReLU aktivasyonu)')
print(f'   • Çıkış katmanı: 3 nöron (Softmax aktivasyonu)')
print('\n3  EĞİTİM AYARLARI:')
print(f'   • Epoch sayısı: 100')
print(f'   • Batch boyutu: 8')
print(f'   • Optimizatör: Adam')
print(f'   • Kayıp fonksiyonu: Categorical Crossentropy')
print(f'   • Başlangıç öğrenme hızı: 0.001')
print('\n4  PERFORMANS:')
print(f'   • Test Doğruluğu: %{test_accuracy * 100:.2f}')
print(f'   • Test Kaybı: {test_loss:.4f}')
print('\n5  BAŞLICA BULGULAR:')
print(f'   • Model Iris çiçeğini {int(test_accuracy * 100)}% doğrulukla sınıflandırıyor')
print(f'   • Eğitim ve validasyon eğrileri yakın (iyi genelleşme)')
print(f'   • En önemli özellik: {iris.feature_names[np.argmax(feature_importance)]}')
print('\n' + ' ' * 20)
print('çalışma başarıyla tamamlandı!')
print(' ' * 20)

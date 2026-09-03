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


# Bölüm 1: Gerekli Kütüphaneleri İçe Aktarma

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import warnings
import time
warnings.filterwarnings('ignore')
print(f"GPU Kullanılabilir: {tf.config.list_physical_devices('GPU')}")
print(f'TensorFlow Sürümü: {tf.__version__}')


# Veri Seti Yükleme ve Keşfetme

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(f'Eğitim veri seti şekli: {X_train.shape}')
print(f'Eğitim etiketleri şekli: {y_train.shape}')
print(f'Test veri seti şekli: {X_test.shape}')
print(f'Test etiketleri şekli: {y_test.shape}')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
y_train = y_train.flatten()
y_test = y_test.flatten()
print(f'\nSınıf sayısı: {len(np.unique(y_train))}')
print(f'Sınıflar: {class_names}')


# Veri Seti Görselleştirme

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()
random_indices = np.random.choice(len(X_train), 10, replace=False)
for idx, ax in enumerate(axes):
    image = X_train[random_indices[idx]]
    label = y_train[random_indices[idx]]
    ax.imshow(image.astype('uint8'))
    ax.set_title(f'Sınıf: {class_names[label]}')
    ax.axis('off')
plt.tight_layout()
save_figure('cifar10_sample_images.png')
print('Örnek görüntüler başarıyla gösterildi.')


# Veri Ön İşleme

X_train_normalized = X_train.astype('float32') / 255.0
X_test_normalized = X_test.astype('float32') / 255.0
num_classes = len(class_names)
y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes)
print(f'Normalize edilmiş eğitim verisi min/max: {X_train_normalized.min():.2f} / {X_train_normalized.max():.2f}')
print(f'One-hot kodlanmış etiket örneği: {y_train_categorical[0]}')
print(f'One-hot kodlanmış etiket şekli: {y_train_categorical.shape}')


# Veri Augmentasyon

train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2, fill_mode='nearest')
test_datagen = ImageDataGenerator()
print('Veri Augmentasyon başarıyla tanımlandı.')


# CNN Modeli Oluşturma

model_cnn = models.Sequential([layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)), layers.BatchNormalization(), layers.Conv2D(32, (3, 3), activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D((2, 2)), layers.Dropout(0.25), layers.Conv2D(64, (3, 3), activation='relu', padding='same'), layers.BatchNormalization(), layers.Conv2D(64, (3, 3), activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D((2, 2)), layers.Dropout(0.25), layers.Conv2D(128, (3, 3), activation='relu', padding='same'), layers.BatchNormalization(), layers.Conv2D(128, (3, 3), activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D((2, 2)), layers.Dropout(0.25), layers.Flatten(), layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5), layers.Dense(num_classes, activation='softmax')])
model_cnn.summary()


# Model Derleme ve Eğitimi

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('Model başarıyla derlendi.')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-07)
print('Model eğitimi başlıyor...')
start_time = time.time()
history = model_cnn.fit(X_train_normalized, y_train_categorical, batch_size=128, epochs=30, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)
training_time = time.time() - start_time
print(f'\nEğitim süresi: {training_time:.2f} saniye')


# Eğitim Sonuçlarını Görselleştirme

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True)
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
save_figure('cnn_training_curves.png')
final_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f'\nFinal Training Accuracy: {final_accuracy:.4f}')
print(f'Final Validation Accuracy: {final_val_accuracy:.4f}')


# Test Seti ile Değerlendirme

test_loss, test_accuracy = model_cnn.evaluate(X_test_normalized, y_test_categorical, verbose=0)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Set Başarısı: {test_accuracy * 100:.2f}%')


# Model Tahminleri ve Sonuçlar

y_pred_probs = model_cnn.predict(X_test_normalized, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.ravel()
random_indices = np.random.choice(len(X_test), 20, replace=False)
for idx, ax in enumerate(axes):
    image = X_test[random_indices[idx]]
    true_label = y_test[random_indices[idx]]
    pred_label = y_pred[random_indices[idx]]
    confidence = y_pred_probs[random_indices[idx]][pred_label]
    ax.imshow(image.astype('uint8'))
    color = 'green' if true_label == pred_label else 'red'
    title = f'Tahmin: {class_names[pred_label]}\n'
    title += f'Gerçek: {class_names[true_label]}\n'
    title += f'Güven: {confidence:.2f}'
    ax.set_title(title, color=color, fontsize=9)
    ax.axis('off')
plt.tight_layout()
save_figure('cnn_test_predictions.png')


# Confusion Matrix ve Sınıflandırma Raporu

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Sınıfı')
plt.tight_layout()
save_figure('cnn_confusion_matrix.png')

report = classification_report(y_test, y_pred, target_names=class_names)
print(report)


# Bölüm 3: Transfer Learning - MobileNetV2 ile CNN

base_model = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
base_model.trainable = False
model_transfer = models.Sequential([base_model, layers.GlobalAveragePooling2D(), layers.Dense(256, activation='relu'), layers.Dropout(0.5), layers.Dense(num_classes, activation='softmax')])
model_transfer.summary()

model_transfer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('Transfer Learning modeli derlendi.')

print('Transfer Learning modeli eğitiliyor...')
start_time = time.time()
history_transfer = model_transfer.fit(X_train_normalized, y_train_categorical, batch_size=128, epochs=15, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)
training_time_transfer = time.time() - start_time
print(f'\nTransfer Learning eğitim süresi: {training_time_transfer:.2f} saniye')

test_loss_transfer, test_accuracy_transfer = model_transfer.evaluate(X_test_normalized, y_test_categorical, verbose=0)
print(f'Transfer Learning Test Accuracy: {test_accuracy_transfer:.4f}')
print(f'Transfer Learning Test Set Başarısı: {test_accuracy_transfer * 100:.2f}%')
print(f'\n--- Model Karşılaştırması ---')
print(f'CNN Model Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)')
print(f'Transfer Learning Test Accuracy: {test_accuracy_transfer:.4f} ({test_accuracy_transfer * 100:.2f}%)')
print(f'Transfer Learning Avantajı: {(test_accuracy_transfer - test_accuracy) * 100:.2f}%')


# YOLO Kurulumu ve Veri Seti Hazırlama

print("YOLO için aşağıdaki komutu terminal'de çalıştırın:")
print('pip install ultralytics')
print('\nYOLO modelleri:')
print('- YOLOv8n (nano): En hızlı, en düşük doğruluk')
print('- YOLOv8s (small): Dengeli')
print('- YOLOv8m (medium): Daha yüksek doğruluk')
print('- YOLOv8l (large): Yüksek doğruluk')
print('- YOLOv8x (xlarge): En yüksek doğruluk, en yavaş')

try:
    from ultralytics import YOLO
    model_yolo = YOLO('yolov8n.pt')
    print('YOLO modeli başarıyla yüklendi.')
    print(f'Model: {model_yolo.model}')
except ImportError:
    print(' ultralytics kurulu değil. Kurulum için:')
    print('pip install ultralytics')
    print('\nUltralytics kurulduktan sonra YOLO bölümü çalışır.')


# YOLO ile Inference (Tahmin Yapma)

try:
    from ultralytics import YOLO
    test_image = X_test[0]
    test_image_bgr = cv2.cvtColor(test_image.astype('uint8'), cv2.COLOR_RGB2BGR)
    model_yolo = YOLO('yolov8n.pt')
    results = model_yolo.predict(test_image_bgr, conf=0.25, verbose=False)
    result = results[0]
    print(f'Algılanan nesne sayısı: {len(result.boxes)}')
    print(f'\nAlgılanan nesneler:')
    for i, box in enumerate(result.boxes):
        coords = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        print(f'Nesne {i + 1}: Sınıf={cls}, Güven={conf:.2f}')
    annotated_frame = result.plot()
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_frame_rgb)
    plt.title('YOLO Object Detection Sonucu')
    plt.axis('off')
    save_figure('yolo_detection_result.png')
except ImportError:
    print('YOLO örneği için ultralytics kurulu olmalıdır.')
    print('pip install ultralytics')


# YOLO Model Bilgileri

coco_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'cat', 15: 'dog', 16: 'horse', 17: 'sheep', 18: 'cow', 19: 'elephant', 20: 'bear', 21: 'zebra', 22: 'giraffe', 23: 'backpack', 24: 'umbrella', 25: 'handbag', 26: 'tie', 27: 'suitcase', 28: 'frisbee', 29: 'skis', 30: 'snowboard', 31: 'sports ball', 32: 'kite', 33: 'baseball bat', 34: 'baseball glove', 35: 'skateboard', 36: 'surfboard', 37: 'tennis racket', 38: 'bottle', 39: 'wine glass', 40: 'cup', 41: 'fork', 42: 'knife', 43: 'spoon', 44: 'bowl', 45: 'banana', 46: 'apple', 47: 'sandwich', 48: 'orange', 49: 'broccoli', 50: 'carrot', 51: 'hot dog', 52: 'pizza', 53: 'donut', 54: 'cake', 55: 'chair', 56: 'sofa', 57: 'pottedplant', 58: 'bed', 59: 'diningtable', 60: 'toilet', 61: 'tvmonitor', 62: 'laptop', 63: 'mouse', 64: 'remote', 65: 'keyboard', 66: 'microwave', 67: 'oven', 68: 'toaster', 69: 'sink', 70: 'refrigerator', 71: 'book', 72: 'clock', 73: 'vase', 74: 'scissors', 75: 'teddy bear', 76: 'hair drier', 77: 'toothbrush'}
print(f'COCO veri seti toplam sınıf sayısı: {len(coco_classes)}')
print(f'\nİlk 20 sınıf:')
for i in range(20):
    print(f'{i}: {coco_classes[i]}')


# Bölüm 5: Özet ve Karşılaştırma

comparison_data = {'Model': ['CNN (Scratch)', 'Transfer Learning (MobileNetV2)'], 'Test Accuracy': [f'{test_accuracy:.4f} ({test_accuracy * 100:.2f}%)', f'{test_accuracy_transfer:.4f} ({test_accuracy_transfer * 100:.2f}%)'], 'Eğitim Süresi': [f'{training_time:.2f}s', f'{training_time_transfer:.2f}s'], 'Epoch Sayısı': [len(history.history['accuracy']), len(history_transfer.history['accuracy'])], 'Parametre Sayısı': [f'{model_cnn.count_params():,}', f'{model_transfer.count_params():,}']}
df_comparison = pd.DataFrame(comparison_data)
print('\n' + '=' * 70)
print('CNN vs TRANSFER LEARNING KARŞILAŞTIRMASI')
print('=' * 70)
print(df_comparison.to_string(index=False))
print('=' * 70)

summary = "\n=== ÖZET ===\n\n1. CNN (Convolutional Neural Networks):\n   - Görüntü sınıflandırması için optimize edilmiş derin öğrenme modeli\n   - Conv2D katmanları: Özellikleri (features) çıkarır\n   - MaxPooling: Boyutu azaltır ve önemli bilgiyi korur\n   - Fully Connected Katmanlar: Sınıflandırma yapar\n   - CIFAR-10'de başarı: {:.2f}%\n\n2. Transfer Learning:\n   - Önceden eğitilmiş modeli yeni görevde kullanma\n   - ImageNet'ten öğrenilen özellikleri yeniden kullanır\n   - Daha az veri ve zaman gerektirir\n   - MobileNetV2 ile başarı: {:.2f}%\n   - Avantaj: {:.2f}% daha iyi performans, {:.1f} kat daha hızlı\n\n3. YOLO (Object Detection):\n   - Gerçek zamanlı nesne algılama\n   - Tek geçişte birden fazla nesneyi algılar\n   - Hızlı ve doğru tahminler yapabilir\n   - COCO veri seti: 80 sınıf\n\n4. Veri Augmentasyon:\n   - Döndürme, kayma, çevirme gibi dönüşümler\n   - Overfitting'i azaltır\n   - Modelin genelleştirme yeteneğini artırır\n\n5. Batch Normalization:\n   - Her katmanda girişi normalize eder\n   - Eğitimi hızlandırır\n   - Daha yüksek öğrenme oranı kullanmayı sağlar\n\n6. Dropout:\n   - Rastgele nöronları devre dışı bırakır\n   - Overfitting'i önler\n   - Daha robust model oluşturur\n".format(test_accuracy * 100, test_accuracy_transfer * 100, (test_accuracy_transfer - test_accuracy) * 100, training_time / training_time_transfer)
print(summary)

print('Modelleri kaydetmek için aşağıdaki kodu çalıştırın:\n')
print('# CNN modeli kaydet')
print("model_cnn.save('cifar10_cnn_model.h5')")
print('')
print('# Transfer Learning modeli kaydet')
print("model_transfer.save('cifar10_transfer_model.h5')")
print('')
print('# Modeli TensorFlow SavedModel formatında kaydet')
print("model_cnn.save('cifar10_cnn_model')")
print('')
print('# Kaydedilen modeli yükle')
print("loaded_model = keras.models.load_model('cifar10_cnn_model.h5')")

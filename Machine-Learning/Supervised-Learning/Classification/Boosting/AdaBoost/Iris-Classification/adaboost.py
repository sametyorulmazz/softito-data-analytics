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


# 2. Zayıf Öğrenci: Karar Kütüğü (Decision Stump)

import numpy as np
import pandas as pd

class DecisionStump:
    """
    En basit karar ağacı: TEK bir özellik ve TEK bir eşik değeri kullanarak
    ikili (+1 / -1) sınıflandırma yapar.
    """

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column <= self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


# 3. En İyi Kütüğü Bulmak: Ağırlıklı Hata (Weighted Error)

def find_best_stump(X, y, weights):
    """
    X: (n_ornek, n_ozellik), y: {-1, +1} etiketleri, weights: her örneğin ağırlığı (toplamı 1 olmalı)
    Tüm özellik/eşik/polarite kombinasyonlarını dener, AĞIRLIKLI HATASI en düşük olanı seçer.
    """
    n_samples, n_features = X.shape
    best_stump = DecisionStump()
    min_error = float('inf')
    for feature_idx in range(n_features):
        X_column = X[:, feature_idx]
        thresholds = np.unique(X_column)
        for threshold in thresholds:
            for polarity in [1, -1]:
                predictions = np.ones(n_samples)
                if polarity == 1:
                    predictions[X_column <= threshold] = -1
                else:
                    predictions[X_column > threshold] = -1
                wrong = predictions != y
                weighted_error = np.sum(weights[wrong])
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_stump.feature_idx = feature_idx
                    best_stump.threshold = threshold
                    best_stump.polarity = polarity
    return (best_stump, min_error)


# 4. Öğrenci Ağırlığı: Alpha ($\alpha$) Hesabı

def compute_alpha(weighted_error, eps=1e-10):
    """
    weighted_error: kütüğün ağırlıklı hatası (0 ile 1 arası)
    eps: err=0 ya da err=1 durumunda log(0) hatasını önlemek için küçük bir sayı
    """
    err = np.clip(weighted_error, eps, 1 - eps)
    alpha = 0.5 * np.log((1 - err) / err)
    return alpha

for err in [0.01, 0.1, 0.3, 0.5, 0.7, 0.99]:
    print(f'Ağırlıklı hata = {err:.2f}  ->  alpha = {compute_alpha(err):+.3f}')


# 5. Örnek Ağırlıklarının Güncellenmesi

def update_weights(weights, alpha, y, predictions):
    """
    weights     : güncellenmeden önceki ağırlıklar
    alpha       : bu turun kütük ağırlığı (söz hakkı)
    y           : gerçek etiketler (-1/+1)
    predictions : bu kütüğün tahminleri (-1/+1)
    """
    new_weights = weights * np.exp(-alpha * y * predictions)
    new_weights = new_weights / np.sum(new_weights)
    return new_weights


# 6. AdaBoost Sınıfı — Her Şeyi Bir Araya Getirmek

class AdaBoostScratch:
    """
    Sıfırdan yazılmış, ikili (+1/-1) sınıflandırma için klasik (discrete) AdaBoost.
    scikit-learn'deki AdaBoostClassifier'ın basitleştirilmiş bir versiyonudur.
    """

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.weight_history = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        weights = np.full(n_samples, 1 / n_samples)
        self.stumps = []
        self.weight_history = [weights.copy()]
        for t in range(self.n_estimators):
            stump, weighted_error = find_best_stump(X, y, weights)
            alpha = compute_alpha(weighted_error)
            stump.alpha = alpha
            predictions = stump.predict(X)
            weights = update_weights(weights, alpha, y, predictions)
            self.stumps.append(stump)
            self.weight_history.append(weights.copy())
        return self

    def predict(self, X):
        X = np.array(X)
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.stumps])
        weighted_sum = np.sum(stump_preds, axis=0)
        return np.sign(weighted_sum)


# 7. Örnek Veri Seti Üzerinde Deneme

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
mask = iris.target != 0
X = iris.data[mask]
y_raw = iris.target[mask]
y = np.where(y_raw == 1, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print('Eğitim seti boyutu:', X_train.shape)
print('Test seti boyutu   :', X_test.shape)
print('Sınıf dağılımı (eğitim):', dict(zip(*np.unique(y_train, return_counts=True))))

ada = AdaBoostScratch(n_estimators=30)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f'AdaBoost (sıfırdan) doğruluğu: {accuracy:.3f}')
print('\nİlk 5 kütüğün özellik/eşik/alpha değerleri:')
for i, stump in enumerate(ada.stumps[:5]):
    fname = iris.feature_names[stump.feature_idx]
    print(f'  Kütük {i + 1}: {fname} (eşik={stump.threshold:.2f}, polarite={stump.polarity:+d}, alpha={stump.alpha:.3f})')


# 8. Örnek Ağırlıklarının İterasyonlar Boyunca Evrimi

import matplotlib.pyplot as plt
weight_history = np.array(ada.weight_history)
weight_increase = weight_history[-1] - weight_history[0]
hardest_samples = np.argsort(weight_increase)[::-1][:5]
plt.figure(figsize=(9, 5))
for idx in hardest_samples:
    plt.plot(weight_history[:, idx], marker='o', markersize=3, label=f'Örnek #{idx} (gerçek sınıf={y_train[idx]:+d})')
plt.axhline(1 / len(y_train), color='gray', linestyle='--', label='Başlangıç ağırlığı (eşit)')
plt.xlabel('Boosting İterasyonu')
plt.ylabel('Örnek Ağırlığı')
plt.title("En 'Zor' Örneklerin Ağırlığı Nasıl Artıyor?")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
save_figure('adaboost_decision_boundary.png')


# 9. `n_estimators` (Kütük Sayısı) Etkisi

n_estimators_list = [1, 2, 5, 10, 20, 30, 50, 80]
train_accs, test_accs = ([], [])
for n_est in n_estimators_list:
    model = AdaBoostScratch(n_estimators=n_est)
    model.fit(X_train, y_train)
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    train_accs.append(train_acc)
    test_accs.append(test_acc)
plt.figure(figsize=(8, 5))
plt.plot(n_estimators_list, train_accs, marker='o', label='Eğitim doğruluğu')
plt.plot(n_estimators_list, test_accs, marker='o', label='Test doğruluğu')
plt.xlabel('Kütük Sayısı (n_estimators)')
plt.ylabel('Doğruluk')
plt.title('AdaBoost: Kütük Sayısının Etkisi')
plt.legend()
plt.grid(alpha=0.3)
save_figure('adaboost_performance.png')


# 10. `scikit-learn` ile Karşılaştırma

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
sk_ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=30, random_state=42)
sk_ada.fit(X_train, y_train)
sk_pred = sk_ada.predict(X_test)
sk_accuracy = (sk_pred == y_test).mean()
print(f'Sıfırdan yazdığımız AdaBoost doğruluğu : {accuracy:.3f}')
print(f'scikit-learn AdaBoostClassifier doğruluğu: {sk_accuracy:.3f}')

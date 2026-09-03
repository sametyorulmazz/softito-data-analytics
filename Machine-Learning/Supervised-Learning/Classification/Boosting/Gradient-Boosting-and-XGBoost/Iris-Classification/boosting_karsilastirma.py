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


# 2. Basit Bir Örnekle Artıklara Ağaç Kurma

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
rng = np.random.RandomState(0)
X_demo = np.linspace(0, 10, 30).reshape(-1, 1)
y_demo = 2 * X_demo.ravel() + 3 + rng.normal(0, 1.5, size=30)
initial_pred = np.full_like(y_demo, y_demo.mean())
residual_1 = y_demo - initial_pred
tree_1 = DecisionTreeRegressor(max_depth=1, random_state=0)
tree_1.fit(X_demo, residual_1)
pred_1 = tree_1.predict(X_demo)
combined_pred = initial_pred + pred_1
residual_2 = y_demo - combined_pred
print('İlk birkaç gerçek y     :', np.round(y_demo[:5], 2))
print('İlk tahmin (ortalama)    :', np.round(initial_pred[:5], 2))
print('1. tur artıkları         :', np.round(residual_1[:5], 2))
print('1. ağacın düzeltmesi      :', np.round(pred_1[:5], 2))
print('Güncellenmiş tahmin       :', np.round(combined_pred[:5], 2))
print('Kalan (2. tur) artıkları  :', np.round(residual_2[:5], 2))


# 5. İkili Sınıflandırma için Gradient Boosting

def sigmoid(z):
    """Log-odds değerini (- ile + arasında herhangi bir sayı) 0-1 arası olasılığa çevirir."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def log_odds(p, eps=1e-10):
    """Bir olasılığı log-odds değerine çevirir (sigmoid'in tersi)."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# 6. `GradientBoostingScratch` Sınıfı — Her Şeyi Bir Araya Getirmek

class GradientBoostingScratch:
    """
    Sıfırdan yazılmış, ikili sınıflandırma için basit Gradient Boosting.
    scikit-learn'deki GradientBoostingClassifier / XGBoost'un temel mantığının basitleştirilmiş halidir.
    """

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.F0 = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        p0 = y.mean()
        self.F0 = log_odds(p0)
        F = np.full(len(y), self.F0)
        self.trees = []
        for t in range(self.n_estimators):
            p = sigmoid(F)
            residual = y - p
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=t)
            tree.fit(X, residual)
            update = tree.predict(X)
            F = F + self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        """Pozitif sınıfa ait olasılığı döndürür."""
        X = np.array(X)
        F = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
        return sigmoid(F)

    def predict(self, X, threshold=0.5):
        """0.5 eşiğine göre nihai sınıf etiketini (0/1) döndürür."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


# 7. Örnek Veri Seti Üzerinde Deneme

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
mask = iris.target != 0
X = iris.data[mask]
y = (iris.target[mask] == 2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print('Eğitim seti boyutu:', X_train.shape)
print('Test seti boyutu   :', X_test.shape)
print('Sınıf dağılımı (eğitim):', dict(zip(*np.unique(y_train, return_counts=True))))

gb = GradientBoostingScratch(n_estimators=50, learning_rate=0.1, max_depth=2)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f'Gradient Boosting (sıfırdan) doğruluğu: {accuracy:.3f}')
proba = gb.predict_proba(X_test)
print('\nİlk 5 test örneği için tahmin edilen olasılıklar:', np.round(proba[:5], 3))
print('Gerçek etiketler                        :', y_test[:5])


# 8. Eğitim Kaybının İterasyonlar Boyunca Azalması

import matplotlib.pyplot as plt

def log_loss(y_true, p, eps=1e-10):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
gb2 = GradientBoostingScratch(n_estimators=80, learning_rate=0.1, max_depth=2)
X_arr, y_arr = (np.array(X_train), np.array(y_train))
p0 = y_arr.mean()
F = np.full(len(y_arr), log_odds(p0))
losses = [log_loss(y_arr, sigmoid(F))]
trees = []
for t in range(gb2.n_estimators):
    p = sigmoid(F)
    residual = y_arr - p
    tree = DecisionTreeRegressor(max_depth=gb2.max_depth, random_state=t)
    tree.fit(X_arr, residual)
    F = F + gb2.learning_rate * tree.predict(X_arr)
    trees.append(tree)
    losses.append(log_loss(y_arr, sigmoid(F)))
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel('Boosting İterasyonu')
plt.ylabel('Eğitim Log-Loss')
plt.title('Gradient Boosting: Her Ağaç Kaybı Nasıl Azaltıyor?')
plt.grid(alpha=0.3)
save_figure('boosting_metrics.png')


# 9. `n_estimators`, `learning_rate` ve `max_depth` Etkileşimi

learning_rates = [0.01, 0.1, 0.5]
n_estimators_list = [1, 5, 10, 20, 40, 80, 150]
plt.figure(figsize=(9, 5))
for lr in learning_rates:
    test_accs = []
    for n_est in n_estimators_list:
        model = GradientBoostingScratch(n_estimators=n_est, learning_rate=lr, max_depth=2)
        model.fit(X_train, y_train)
        acc = (model.predict(X_test) == y_test).mean()
        test_accs.append(acc)
    plt.plot(n_estimators_list, test_accs, marker='o', label=f'learning_rate={lr}')
plt.xlabel('Ağaç Sayısı (n_estimators)')
plt.ylabel('Test Doğruluğu')
plt.title('Öğrenme Oranı ve Ağaç Sayısının Birlikte Etkisi')
plt.legend()
plt.grid(alpha=0.3)
save_figure('boosting_importance.png')


# 11. Gerçek `xgboost` Kütüphanesi ve `scikit-learn` ile Karşılaştırma

from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
sk_gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42)
sk_gb.fit(X_train, y_train)
sk_gb_acc = (sk_gb.predict(X_test) == y_test).mean()
xgb_model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_acc = (xgb_model.predict(X_test) == y_test).mean()
print(f'Sıfırdan yazdığımız Gradient Boosting doğruluğu : {accuracy:.3f}')
print(f'scikit-learn GradientBoostingClassifier doğruluğu: {sk_gb_acc:.3f}')
print(f'XGBoost (gerçek kütüphane) doğruluğu              : {xgb_acc:.3f}')

importances = xgb_model.feature_importances_
order = np.argsort(importances)[::-1]
plt.figure(figsize=(7, 4))
plt.barh([iris.feature_names[i] for i in order][::-1], importances[order][::-1])
plt.xlabel('Önem Skoru')
plt.title('XGBoost Özellik Önemi')
plt.tight_layout()
save_figure('boosting_timing.png')

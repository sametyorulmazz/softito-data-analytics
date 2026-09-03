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


# 2. Bootstrap Örnekleme (Bagging'in İlk Ayağı)

import numpy as np
import pandas as pd

def bootstrap_sample(X, y, random_state=None):
    """
    X: (n_ornek, n_ozellik) boyutlu özellik matrisi
    y: (n_ornek,) boyutlu etiket dizisi
    random_state: tekrar üretilebilirlik için rastgelelik tohumu (seed)
    Dönüş: (X_boot, y_boot) -- orijinal veriyle AYNI BOYUTTA, yerine koyarak seçilmiş yeni örneklem
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    indices = rng.randint(0, n_samples, size=n_samples)
    return (X[indices], y[indices])

X_demo = np.arange(10).reshape(-1, 1)
y_demo = np.arange(10)
X_boot, y_boot = bootstrap_sample(X_demo, y_demo, random_state=42)
print('Orijinal y :', y_demo)
print('Bootstrap y:', y_boot.ravel())
print('\nBenzersiz (unique) örnek sayısı:', len(np.unique(y_boot)), '/ 10')


# 3. Temel Yapı Taşı: Basit Karar Ağacı

def entropy(y):
    """Bir kümenin belirsizliğini (entropisini) ölçer: -sum(p_i * log2(p_i))"""
    y = np.array(y)
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def information_gain(y, y_left, y_right):
    """Bir bölünmenin ne kadar entropi azalttığını (bilgi kazancını) ölçer."""
    n, n_left, n_right = (len(y), len(y_left), len(y_right))
    if n_left == 0 or n_right == 0:
        return 0.0
    parent = entropy(y)
    weighted_child = n_left / n * entropy(y_left) + n_right / n * entropy(y_right)
    return parent - weighted_child

class Node:
    """Ağaçta tek bir düğüm: iç düğümse feature/threshold, yapraksa value doludur."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

def best_split(X, y, max_features=None, rng=None):
    """
    X, y          : mevcut düğümdeki veri
    max_features  : her bölünmede DENENECEK özellik sayısı (None ise TÜM özellikler denenir)
    rng           : rastgelelik üreteci (hangi özelliklerin seçileceğini belirler)
    Dönüş: (en_iyi_ozellik, en_iyi_esik, en_iyi_kazanc)
    """
    n_samples, n_features = X.shape
    if max_features is None:
        feature_indices = np.arange(n_features)
    else:
        feature_indices = rng.choice(n_features, size=max_features, replace=False)
    best_gain = 0.0
    best_feature, best_threshold = (None, None)
    for feature_idx in feature_indices:
        values = X[:, feature_idx]
        thresholds = np.unique(values)
        for t in thresholds:
            left_mask = values <= t
            right_mask = values > t
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gain = information_gain(y, y[left_mask], y[right_mask])
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = t
    return (best_feature, best_threshold, best_gain)


# 4. Tek Bir Ağacı Kurmak ve Tahmin Üretmek

def build_tree(X, y, depth=0, max_depth=5, min_samples_split=2, max_features=None, rng=None):
    """Bir karar ağacını özyinelemeli olarak kurar ve kök Node nesnesini döndürür."""
    n_samples = len(y)
    n_classes = len(np.unique(y))
    if depth >= max_depth or n_classes == 1 or n_samples < min_samples_split:
        leaf_value = most_common_label(y)
        return Node(value=leaf_value)
    feature, threshold, gain = best_split(X, y, max_features=max_features, rng=rng)
    if feature is None or gain <= 0:
        leaf_value = most_common_label(y)
        return Node(value=leaf_value)
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_split, max_features, rng)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_split, max_features, rng)
    return Node(feature=feature, threshold=threshold, left=left, right=right)

def most_common_label(y):
    """Bir kümedeki en sık geçen (çoğunluk) sınıfı döndürür."""
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]

def predict_one(node, x):
    """Tek bir örneği (x) kökten yaprağa kadar dolaştırarak tahmin üretir."""
    if node.is_leaf():
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)


# 5. Random Forest Sınıfı — Her Şeyi Bir Araya Getirmek

class RandomForestScratch:
    """
    Sıfırdan yazılmış basit bir Random Forest sınıflandırıcısı.
    scikit-learn'deki RandomForestClassifier'ın basitleştirilmiş bir versiyonudur.
    """

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _resolve_max_features(self, n_features):
        """max_features parametresini ("sqrt", "log2", sayı, None) gerçek bir tam sayıya çevirir."""
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]
        mf = self._resolve_max_features(n_features)
        self.trees = []
        master_rng = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            tree_seed = master_rng.randint(0, 1000000)
            tree_rng = np.random.RandomState(tree_seed)
            X_boot, y_boot = bootstrap_sample(X, y, random_state=tree_seed)
            root = build_tree(X_boot, y_boot, depth=0, max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=mf, rng=tree_rng)
            self.trees.append(root)
        return self

    def predict(self, X):
        X = np.array(X)
        all_predictions = np.array([[predict_one(tree, x) for x in X] for tree in self.trees])
        final_predictions = []
        for col in range(all_predictions.shape[1]):
            votes = all_predictions[:, col]
            values, counts = np.unique(votes, return_counts=True)
            final_predictions.append(values[np.argmax(counts)])
        return np.array(final_predictions)

    def predict_proba(self, X):
        """Her sınıf için oy oranını (0-1 arası olasılık gibi) döndürür."""
        X = np.array(X)
        all_predictions = np.array([[predict_one(tree, x) for x in X] for tree in self.trees])
        classes = np.unique(all_predictions)
        probs = np.zeros((X.shape[0], len(classes)))
        for col in range(all_predictions.shape[1]):
            votes = all_predictions[:, col]
            for c_idx, c in enumerate(classes):
                probs[col, c_idx] = np.mean(votes == c)
        return (probs, classes)


# 6. Örnek Veri Seti Üzerinde Deneme

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = (iris.data, iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Eğitim seti boyutu:', X_train.shape)
print('Test seti boyutu   :', X_test.shape)
print('Özellikler:', iris.feature_names)
print('Sınıflar  :', iris.target_names)

rf = RandomForestScratch(n_estimators=20, max_depth=4, max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f'Random Forest (sıfırdan) doğruluğu: {accuracy:.3f}')

single_tree_root = build_tree(X_train, y_train, depth=0, max_depth=4, min_samples_split=2, max_features=None, rng=None)
single_tree_pred = np.array([predict_one(single_tree_root, x) for x in X_test])
single_tree_acc = (single_tree_pred == y_test).mean()
print(f'Tek karar ağacı doğruluğu     : {single_tree_acc:.3f}')
print(f'Random Forest (20 ağaç) doğruluğu: {accuracy:.3f}')


# 7. `n_estimators` (Ağaç Sayısı) Etkisi ve Kararlılık

import matplotlib.pyplot as plt
n_estimators_list = [1, 3, 5, 10, 20, 40]
n_repeats = 8
means, stds = ([], [])
for n_est in n_estimators_list:
    accs = []
    for rep in range(n_repeats):
        model = RandomForestScratch(n_estimators=n_est, max_depth=4, max_features='sqrt', random_state=rep)
        model.fit(X_train, y_train)
        acc = (model.predict(X_test) == y_test).mean()
        accs.append(acc)
    means.append(np.mean(accs))
    stds.append(np.std(accs))
plt.figure(figsize=(8, 5))
plt.errorbar(n_estimators_list, means, yerr=stds, marker='o', capsize=4)
plt.xlabel('Ağaç Sayısı (n_estimators)')
plt.ylabel('Test Doğruluğu (ortalama ± std)')
plt.title('Ağaç Sayısı Arttıkça Kararlılık Artar (Varyans Azalır)')
plt.grid(alpha=0.3)
save_figure('forest_comparison.png')
for n_est, m, s in zip(n_estimators_list, means, stds):
    print(f'n_estimators={n_est:>3} -> ortalama doğruluk={m:.3f}, std={s:.4f}')


# 8. Özellik Önemi (Feature Importance) — Basit Bir Yaklaşım

def permutation_importance(model, X, y, n_repeats=10, random_state=0):
    """
    model     : eğitilmiş RandomForestScratch nesnesi
    X, y      : değerlendirme (örn. test) verisi
    n_repeats : her özellik için karıştırma işleminin kaç kez tekrarlanacağı (gürültüyü azaltmak için)
    Dönüş     : her özellik için ortalama önem skoru (baseline_doğruluk - karıştırılmış_doğruluk)
    """
    rng = np.random.RandomState(random_state)
    baseline_acc = (model.predict(X) == y).mean()
    n_features = X.shape[1]
    importances = np.zeros(n_features)
    for feature_idx in range(n_features):
        drops = []
        for rep in range(n_repeats):
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, feature_idx])
            permuted_acc = (model.predict(X_permuted) == y).mean()
            drops.append(baseline_acc - permuted_acc)
        importances[feature_idx] = np.mean(drops)
    return importances

importances = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=0)
order = np.argsort(importances)[::-1]
print('Özellik Önem Sıralaması:')
for idx in order:
    print(f'  {iris.feature_names[idx]:<20} -> {importances[idx]:.4f}')
plt.figure(figsize=(7, 4))
plt.barh([iris.feature_names[i] for i in order][::-1], importances[order][::-1])
plt.xlabel('Önem Skoru (doğruluk düşüşü)')
plt.title('Permütasyon ile Özellik Önemi')
plt.tight_layout()
save_figure('feature_importance.png')


# 9. `scikit-learn` ile Karşılaştırma

from sklearn.ensemble import RandomForestClassifier
sk_rf = RandomForestClassifier(n_estimators=20, max_depth=4, max_features='sqrt', random_state=42)
sk_rf.fit(X_train, y_train)
sk_pred = sk_rf.predict(X_test)
sk_acc = (sk_pred == y_test).mean()
print(f'Sıfırdan yazdığımız Random Forest doğruluğu : {accuracy:.3f}')
print(f'scikit-learn RandomForestClassifier doğruluğu: {sk_acc:.3f}')
print("\nscikit-learn'ün kendi özellik önemleri (Gini importance):")
for name, imp in sorted(zip(iris.feature_names, sk_rf.feature_importances_), key=lambda t: -t[1]):
    print(f'  {name:<20} -> {imp:.4f}')


# 10. Out-of-Bag (OOB) Skoru — Bonus Kavram

def compute_oob_score(X, y, n_estimators=30, max_depth=4, max_features='sqrt', random_state=42):
    X, y = (np.array(X), np.array(y))
    n_samples, n_features = X.shape
    mf = max(1, int(np.sqrt(n_features))) if max_features == 'sqrt' else n_features
    master_rng = np.random.RandomState(random_state)
    oob_votes = [[] for _ in range(n_samples)]
    for i in range(n_estimators):
        tree_seed = master_rng.randint(0, 1000000)
        tree_rng = np.random.RandomState(tree_seed)
        boot_idx = np.random.RandomState(tree_seed).randint(0, n_samples, size=n_samples)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        X_boot, y_boot = (X[boot_idx], y[boot_idx])
        root = build_tree(X_boot, y_boot, depth=0, max_depth=max_depth, max_features=mf, rng=tree_rng)
        oob_indices = np.where(oob_mask)[0]
        for idx in oob_indices:
            pred = predict_one(root, X[idx])
            oob_votes[idx].append(pred)
    correct = 0
    evaluated = 0
    for idx in range(n_samples):
        if len(oob_votes[idx]) == 0:
            continue
        values, counts = np.unique(oob_votes[idx], return_counts=True)
        oob_pred = values[np.argmax(counts)]
        correct += oob_pred == y[idx]
        evaluated += 1
    return correct / evaluated
oob_acc = compute_oob_score(X_train, y_train, n_estimators=30, random_state=42)
print(f'OOB doğruluğu (ayrı test seti KULLANMADAN tahmin edilen): {oob_acc:.3f}')
print(f'Gerçek test seti doğruluğu (karşılaştırma için)         : {accuracy:.3f}')

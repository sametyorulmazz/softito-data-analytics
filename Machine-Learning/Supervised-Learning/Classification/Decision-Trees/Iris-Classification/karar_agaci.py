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


# 2. Saflık Ölçütü 1: Entropi (Entropy)

import numpy as np
import pandas as pd

def entropy(y):
    """
    y: sınıf etiketlerinin bulunduğu bir dizi (liste, numpy array veya pandas Series)
    Dönüş: kümenin entropisi (0 ile log2(k) arasında bir sayı)
    """
    y = np.array(y)
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

print('Tamamen saf küme  [1,1,1,1]  -> Entropi:', entropy([1, 1, 1, 1]))
print('Tam karışık küme  [0,1,0,1]  -> Entropi:', entropy([0, 1, 0, 1]))
print('Kısmen karışık    [0,0,0,1]  -> Entropi:', entropy([0, 0, 0, 1]))


# 3. Saflık Ölçütü 2: Gini Safsızlığı (Gini Impurity)

def gini(y):
    """
    y: sınıf etiketleri
    Dönüş: Gini safsızlık değeri (0 = tamamen saf, üst sınır sınıf sayısına göre değişir)
    """
    y = np.array(y)
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1 - np.sum(probs ** 2)

print('Tamamen saf küme  [1,1,1,1]  -> Gini:', gini([1, 1, 1, 1]))
print('Tam karışık küme  [0,1,0,1]  -> Gini:', gini([0, 1, 0, 1]))
print('Kısmen karışık    [0,0,0,1]  -> Gini:', gini([0, 0, 0, 1]))


# 4. Bilgi Kazancı (Information Gain) — Bölünmenin Kalitesini Ölçmek

def information_gain(y, y_left, y_right, criterion='entropy'):
    """
    y       : bölünmeden önceki tüm etiketler (üst düğüm)
    y_left  : bölünme sonrası sol dala giden etiketler
    y_right : bölünme sonrası sağ dala giden etiketler
    criterion: "entropy" ya da "gini" -- hangi saflık ölçütü kullanılacak
    """
    impurity_fn = entropy if criterion == 'entropy' else gini
    n = len(y)
    n_left, n_right = (len(y_left), len(y_right))
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = impurity_fn(y)
    weighted_child_impurity = n_left / n * impurity_fn(y_left) + n_right / n * impurity_fn(y_right)
    return parent_impurity - weighted_child_impurity


# 5. En İyi Bölünmeyi Bulmak (Best Split)

def best_split(X, y, criterion='entropy'):
    """
    X: 2 boyutlu numpy array, şekli (n_ornek, n_ozellik)
    y: 1 boyutlu numpy array, etiketler
    Dönüş: (en_iyi_ozellik_indeksi, en_iyi_esik, en_iyi_kazanc)
           Uygun bölünme yoksa (None, None, 0) döner
    """
    n_samples, n_features = X.shape
    best_gain = 0.0
    best_feature = None
    best_threshold = None
    for feature_idx in range(n_features):
        values = X[:, feature_idx]
        thresholds = np.unique(values)
        for t in thresholds:
            left_mask = values <= t
            right_mask = values > t
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gain = information_gain(y, y[left_mask], y[right_mask], criterion=criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = t
    return (best_feature, best_threshold, best_gain)


# 6. Ağaç Düğümü (Node) Sınıfı

class Node:
    """
    Karar ağacındaki tek bir düğümü temsil eder.
    Bir düğüm ya bir 'iç düğüm' (bölünme yapar) ya da bir 'yaprak' (karar verir) olabilir.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


# 7. Karar Ağacını Kurmak: Özyinelemeli (Recursive) Algoritma

class DecisionTreeScratch:
    """
    Sıfırdan yazılmış basit bir sınıflandırma (classification) karar ağacı.
    scikit-learn'deki DecisionTreeClassifier'ın çok basitleştirilmiş bir versiyonudur.
    """

    def __init__(self, max_depth=5, min_samples_split=2, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        feature, threshold, gain = best_split(X, y, self.criterion)
        if feature is None or gain <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

    @staticmethod
    def _most_common_label(y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict_one(self, x, node=None):
        """Tek bir örnek (x) için ağacı kökten yaprağa kadar dolaşarak tahmin üretir."""
        if node is None:
            node = self.root
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        """Birden fazla örnek için predict_one'ı tek tek çağırır."""
        X = np.array(X)
        return np.array([self.predict_one(x) for x in X])


# 8. Örnek Veri Seti Üzerinde Deneme

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = (iris.data, iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Eğitim seti boyutu:', X_train.shape)
print('Test seti boyutu   :', X_test.shape)
print('Sınıflar:', iris.target_names)

tree = DecisionTreeScratch(max_depth=3, min_samples_split=2, criterion='entropy')
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f'Sıfırdan yazdığımız ağacın doğruluğu: {accuracy:.3f}')


# 9. Kurulan Ağacı Yazdırmak

def print_tree(node, feature_names, class_names, depth=0):
    """
    node          : yazdırılacak (alt) ağacın kök düğümü
    feature_names : sütun isimleri (örn. iris.feature_names)
    class_names   : sınıf isimleri (örn. iris.target_names)
    depth         : mevcut girinti seviyesi (özyineleme sırasında artar)
    """
    indent = '  ' * depth
    if node.is_leaf():
        print(f'{indent}-> Tahmin: {class_names[node.value]}')
        return
    feature_name = feature_names[node.feature]
    print(f'{indent}[{feature_name} <= {node.threshold:.2f}]')
    print(f'{indent}Evet ise:')
    print_tree(node.left, feature_names, class_names, depth + 1)
    print(f'{indent}Hayır ise:')
    print_tree(node.right, feature_names, class_names, depth + 1)
print_tree(tree.root, iris.feature_names, iris.target_names)


# 10. Grafiksel Görselleştirme

import matplotlib.pyplot as plt

def plot_node(ax, node, x, y, dx, feature_names, class_names, depth=0, max_depth=3):
    """
    Ağacı matplotlib ekseninde (ax) özyinelemeli olarak çizer.
    x, y   : bu düğümün ekrandaki konumu
    dx     : bir sonraki seviyede sol/sağ çocuklar arasındaki yatay mesafe
    """
    if node.is_leaf():
        text = f'{class_names[node.value]}'
        ax.text(x, y, text, ha='center', va='center', bbox=dict(boxstyle='round', fc='lightgreen'))
        return
    feature_name = feature_names[node.feature]
    text = f'{feature_name}\n<= {node.threshold:.2f}'
    ax.text(x, y, text, ha='center', va='center', bbox=dict(boxstyle='round', fc='lightblue'))
    y_child = y - 1
    x_left, x_right = (x - dx, x + dx)
    ax.plot([x, x_left], [y - 0.1, y_child + 0.1], 'k-')
    ax.plot([x, x_right], [y - 0.1, y_child + 0.1], 'k-')
    plot_node(ax, node.left, x_left, y_child, dx / 2, feature_names, class_names, depth + 1, max_depth)
    plot_node(ax, node.right, x_right, y_child, dx / 2, feature_names, class_names, depth + 1, max_depth)
fig, ax = plt.subplots(figsize=(12, 6))
plot_node(ax, tree.root, x=0, y=0, dx=4, feature_names=iris.feature_names, class_names=iris.target_names)
ax.axis('off')
plt.title('Sıfırdan Yazılan Karar Ağacı (max_depth=3)')
plt.tight_layout()
save_figure('decision_boundary.png')


# 11. `scikit-learn` ile Karşılaştırma

from sklearn.tree import DecisionTreeClassifier, plot_tree
sk_tree = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
sk_tree.fit(X_train, y_train)
sk_pred = sk_tree.predict(X_test)
sk_accuracy = (sk_pred == y_test).mean()
print(f'Sıfırdan yazdığımız ağaç doğruluğu : {accuracy:.3f}')
print(f'scikit-learn ağacı doğruluğu       : {sk_accuracy:.3f}')

plt.figure(figsize=(14, 8))
plot_tree(sk_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title('scikit-learn DecisionTreeClassifier (max_depth=3)')
save_figure('tree_structure.png')


# 12. Aşırı Öğrenme (Overfitting) ve `max_depth`'in Etkisi

depths = range(1, 11)
train_accs = []
test_accs = []
for d in depths:
    t = DecisionTreeScratch(max_depth=d, criterion='entropy')
    t.fit(X_train, y_train)
    train_acc = (t.predict(X_train) == y_train).mean()
    test_acc = (t.predict(X_test) == y_test).mean()
    train_accs.append(train_acc)
    test_accs.append(test_acc)
plt.figure(figsize=(8, 5))
plt.plot(depths, train_accs, marker='o', label='Eğitim doğruluğu')
plt.plot(depths, test_accs, marker='o', label='Test doğruluğu')
plt.xlabel('max_depth')
plt.ylabel('Doğruluk')
plt.title('Ağaç Derinliği ile Aşırı Öğrenme İlişkisi')
plt.legend()
plt.grid(alpha=0.3)
save_figure('depth_comparison.png')

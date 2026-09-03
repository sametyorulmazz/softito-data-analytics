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


# Deney tasarımı

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
SEED = 42
N_REPEAT = 40

def run_experiment(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 2000))
    y = rng.integers(0, 2, size=120)
    selector = SelectKBest(f_classif, k=20)
    X_leaky = selector.fit_transform(X, y)
    Xtr_w, Xte_w, ytr_w, yte_w = train_test_split(X_leaky, y, test_size=0.3, stratify=y, random_state=seed)
    wrong = LogisticRegression(max_iter=2000).fit(Xtr_w, ytr_w)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    right = Pipeline([('select', SelectKBest(f_classif, k=20)), ('model', LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
    return (accuracy_score(yte_w, wrong.predict(Xte_w)), accuracy_score(yte, right.predict(Xte)))
scores = np.array([run_experiment(SEED + i) for i in range(N_REPEAT)])
results = pd.DataFrame(scores, columns=['Sızıntılı akış', 'Pipeline akışı'])
results.describe().loc[['mean', 'std', 'min', 'max']].round(3)

fig, ax = plt.subplots(figsize=(7, 4))
results.boxplot(ax=ax, grid=False)
ax.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Rastgele tahmin')
ax.set_ylabel('Test doğruluğu')
ax.set_title('Test Etiketlerini Gören Özellik Seçimi Yanıltır')
ax.legend()
plt.tight_layout()
save_figure('data_leakage_comparison.png')

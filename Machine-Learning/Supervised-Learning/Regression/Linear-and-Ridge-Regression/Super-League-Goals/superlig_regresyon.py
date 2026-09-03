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


# Amaç ve kapsam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Veri kontrolü

df = pd.read_excel('data/superlig_proje.xlsx')
target = 'Atilan_Gol'
print(df[['Takim_Adi', 'xG_Gol_Beklentisi', target]].head())
print('Boyut:', df.shape)
print('Eksik değer:', int(df.isna().sum().sum()))


# 2. Önceden belirlenen modeller

feature_sets = {'Yalnızca xG': ['xG_Gol_Beklentisi'], 'Ridge (3 hücum göstergesi)': ['xG_Gol_Beklentisi', 'Isabetli_Sut', 'Buyuk_Sans_Yaratma']}
models = {'Yalnızca xG': Pipeline([('imputer', SimpleImputer()), ('model', LinearRegression())]), 'Ridge (3 hücum göstergesi)': Pipeline([('imputer', SimpleImputer()), ('scale', StandardScaler()), ('model', Ridge(alpha=10.0))])}
loo = LeaveOneOut()
rows, predictions = ([], {})
for name, features in feature_sets.items():
    pred = cross_val_predict(models[name], df[features], df[target], cv=loo)
    predictions[name] = pred
    rows.append({'Model': name, 'LOOCV MAE': mean_absolute_error(df[target], pred), 'LOOCV R2': r2_score(df[target], pred)})
metrics = pd.DataFrame(rows).set_index('Model')
metrics.round(3)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
for ax, (name, pred) in zip(axes, predictions.items()):
    ax.scatter(df[target], pred, s=55, alpha=0.8)
    lo = min(df[target].min(), pred.min())
    hi = max(df[target].max(), pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    for team, actual, estimate in zip(df['Takim_Adi'], df[target], pred):
        if abs(actual - estimate) >= np.quantile(abs(df[target] - pred), 0.8):
            ax.annotate(team, (actual, estimate), fontsize=7)
    ax.set_title(name)
    ax.set_xlabel('Gerçek gol')
axes[0].set_ylabel('LOOCV tahmini')
plt.tight_layout()
save_figure('superleague_predictions.png')

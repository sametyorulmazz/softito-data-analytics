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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
n = 5000
need = rng.normal(size=n)
propensity_true = 1 / (1 + np.exp(-(-0.2 + 1.0 * need)))
treatment = rng.binomial(1, propensity_true)
true_effect = 3.0
outcome = 10 + 4 * need + true_effect * treatment + rng.normal(scale=2, size=n)
df = pd.DataFrame({'need': need, 'treatment': treatment, 'outcome': outcome})
df.groupby('treatment')[['need', 'outcome']].mean().round(3)

naive_ate = df.loc[df.treatment.eq(1), 'outcome'].mean() - df.loc[df.treatment.eq(0), 'outcome'].mean()
reg = LinearRegression().fit(df[['treatment', 'need']], df['outcome'])
regression_ate = reg.coef_[0]
ps_model = LogisticRegression().fit(df[['need']], df['treatment'])
ps = np.clip(ps_model.predict_proba(df[['need']])[:, 1], 0.02, 0.98)
treated_mean = np.sum(df.treatment * df.outcome / ps) / np.sum(df.treatment / ps)
control_mean = np.sum((1 - df.treatment) * df.outcome / (1 - ps)) / np.sum((1 - df.treatment) / (1 - ps))
ipw_ate = treated_mean - control_mean
pd.Series({'Gerçek etki': true_effect, 'Naif fark': naive_ate, 'Regresyon düzeltmesi': regression_ate, 'IPW': ipw_ate}).round(3)

estimates = pd.Series({'Gerçek': true_effect, 'Naif': naive_ate, 'Regresyon': regression_ate, 'IPW': ipw_ate})
fig, ax = plt.subplots(figsize=(7, 4))
estimates.plot.bar(ax=ax, color=['black', '#C44E52', '#4C72B0', '#55A868'])
ax.axhline(true_effect, linestyle='--', color='black', linewidth=1)
ax.set_ylabel('Tahmini ortalama tedavi etkisi')
ax.set_title('Karıştırıcı Değişkenin Düzeltilmesi')
plt.xticks(rotation=0)
plt.tight_layout()
save_figure('causal_effect_comparison.png')

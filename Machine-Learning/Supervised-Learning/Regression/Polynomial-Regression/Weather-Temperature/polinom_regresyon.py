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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('data/weather_summary.csv', low_memory=False, usecols=['STA', 'Date', 'MaxTemp', 'MeanTemp'])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
for col in ['MaxTemp', 'MeanTemp']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().sort_values(['Date', 'STA']).reset_index(drop=True)
print('Boyut:', df.shape)
print('Tarih aralığı:', df['Date'].min().date(), '-', df['Date'].max().date())
df.head()


# Ayrım ve model seçimi

cut = int(len(df) * 0.8)
train, test = (df.iloc[:cut], df.iloc[cut:])
X_train, y_train = (train[['MaxTemp']], train['MeanTemp'])
X_test, y_test = (test[['MaxTemp']], test['MeanTemp'])
degrees = [1, 2, 3, 5]
tscv = TimeSeriesSplit(n_splits=5)
rows = []
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')
    rows.append({'degree': degree, 'cv_rmse_mean': -scores.mean(), 'cv_rmse_std': scores.std()})
cv_results = pd.DataFrame(rows)
best_degree = int(cv_results.loc[cv_results['cv_rmse_mean'].idxmin(), 'degree'])
print('Seçilen derece:', best_degree)
cv_results.round(3)

model = make_pipeline(PolynomialFeatures(degree=best_degree, include_bias=False), LinearRegression())
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = pd.Series({'Test MAE': mean_absolute_error(y_test, pred), 'Test RMSE': mean_squared_error(y_test, pred) ** 0.5, 'Test R2': r2_score(y_test, pred)})
metrics.round(3)

rng = np.random.default_rng(42)
idx = rng.choice(len(test), size=min(3000, len(test)), replace=False)
x_line = np.linspace(df['MaxTemp'].quantile(0.01), df['MaxTemp'].quantile(0.99), 300).reshape(-1, 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(X_test.iloc[idx, 0], y_test.iloc[idx], s=8, alpha=0.25)
axes[0].plot(x_line, model.predict(x_line), color='#C44E52', linewidth=2)
axes[0].set(title=f'Polinom Derecesi: {best_degree}', xlabel='MaxTemp', ylabel='MeanTemp')
residual = y_test.to_numpy() - pred
axes[1].scatter(pred[idx], residual[idx], s=8, alpha=0.25)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set(title='Test Artıkları', xlabel='Tahmin', ylabel='Gerçek - tahmin')
plt.tight_layout()
save_figure('polynomial_regression.png')

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


# 1. Gerekli Kütüphanelerin Yüklenmesi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
np.random.seed(42)


# 2. Veri Setinin Yüklenmesi ve İncelenmesi

df = pd.read_csv(DATA_DIR / 'diabetes_regression.csv')
X = df.drop(columns='hastalik_ilerlemesi')
y = df['hastalik_ilerlemesi']
print('Özellik sayısı:', X.shape[1])
print('Gözlem sayısı :', X.shape[0])
X.head()

y.describe()


# 3. Eğitim / Test Ayrımı ve Özellik Ölçekleme

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Eğitim seti boyutu:', X_train.shape)
print('Test seti boyutu  :', X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
X_train_scaled.head()


# 4. Ridge Regresyon (L2 Regularizasyon)

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)
print(f'Ridge  -> MSE: {ridge_mse:.2f} | MAE: {ridge_mae:.2f} | R2: {ridge_r2:.4f}')


# 5. Lasso Regresyon (L1 Regularizasyon)

lasso_model = Lasso(alpha=0.1, random_state=42)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)
print(f'Lasso  -> MSE: {lasso_mse:.2f} | MAE: {lasso_mae:.2f} | R2: {lasso_r2:.4f}')
n_sifir_katsayi = np.sum(lasso_model.coef_ == 0)
print(f'Lasso tarafından sıfırlanan özellik sayısı: {n_sifir_katsayi} / {len(lasso_model.coef_)}')


# 6. Elastic Net Regresyon (L1 + L2 Kombinasyonu)

elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_model.fit(X_train_scaled, y_train)
y_pred_elastic = elastic_model.predict(X_test_scaled)
elastic_mse = mean_squared_error(y_test, y_pred_elastic)
elastic_mae = mean_absolute_error(y_test, y_pred_elastic)
elastic_r2 = r2_score(y_test, y_pred_elastic)
print(f'ElasticNet -> MSE: {elastic_mse:.2f} | MAE: {elastic_mae:.2f} | R2: {elastic_r2:.4f}')


# 7. Model Karşılaştırması

sonuclar = pd.DataFrame({'Model': ['Ridge', 'Lasso', 'ElasticNet'], 'MSE': [ridge_mse, lasso_mse, elastic_mse], 'MAE': [ridge_mae, lasso_mae, elastic_mae], 'R2': [ridge_r2, lasso_r2, elastic_r2]})
sonuclar = sonuclar.sort_values('R2', ascending=False)
sonuclar

katsayilar = pd.DataFrame({'Özellik': X.columns, 'Ridge': ridge_model.coef_, 'Lasso': lasso_model.coef_, 'ElasticNet': elastic_model.coef_})
katsayilar

fig, ax = plt.subplots(figsize=(10, 5))
genislik = 0.25
x_konum = np.arange(len(X.columns))
ax.bar(x_konum - genislik, katsayilar['Ridge'], width=genislik, label='Ridge')
ax.bar(x_konum, katsayilar['Lasso'], width=genislik, label='Lasso')
ax.bar(x_konum + genislik, katsayilar['ElasticNet'], width=genislik, label='ElasticNet')
ax.set_xticks(x_konum)
ax.set_xticklabels(X.columns, rotation=45, ha='right')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Katsayı Değeri')
ax.set_title('Ridge, Lasso ve ElasticNet Katsayı Karşılaştırması')
ax.legend()
plt.tight_layout()
save_figure('regularization_comparison.png')


# 8. GridSearchCV ile Hiperparametre Optimizasyonu

alpha_araligi = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_grid = GridSearchCV(Ridge(random_state=42), param_grid=alpha_araligi, cv=5, scoring='r2')
ridge_grid.fit(X_train_scaled, y_train)
lasso_grid = GridSearchCV(Lasso(random_state=42, max_iter=10000), param_grid=alpha_araligi, cv=5, scoring='r2')
lasso_grid.fit(X_train_scaled, y_train)
elastic_araligi = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
elastic_grid = GridSearchCV(ElasticNet(random_state=42, max_iter=10000), param_grid=elastic_araligi, cv=5, scoring='r2')
elastic_grid.fit(X_train_scaled, y_train)
print('En iyi Ridge alpha       :', ridge_grid.best_params_)
print('En iyi Lasso alpha       :', lasso_grid.best_params_)
print('En iyi ElasticNet params :', elastic_grid.best_params_)

en_iyi_ridge = ridge_grid.best_estimator_
en_iyi_lasso = lasso_grid.best_estimator_
en_iyi_elastic = elastic_grid.best_estimator_
for isim, model in [('Ridge (optimize)', en_iyi_ridge), ('Lasso (optimize)', en_iyi_lasso), ('ElasticNet (optimize)', en_iyi_elastic)]:
    tahmin = model.predict(X_test_scaled)
    r2 = r2_score(y_test, tahmin)
    mse = mean_squared_error(y_test, tahmin)
    print(f'{isim:22s} -> R2: {r2:.4f} | MSE: {mse:.2f}')

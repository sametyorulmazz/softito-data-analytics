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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from sklearn.linear_model import RidgeCV
np.random.seed(42)


# 2. Veri Setinin Yüklenmesi ve İncelenmesi

df = pd.read_csv(DATA_DIR / 'diamonds.csv')

print(df.isna().sum())
print((df[['x', 'y', 'z']] == 0).sum())

df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
df = df.dropna()
df = df.reset_index(drop=True)
print('Temizlenen:', df.shape)

print(df.isna().sum())
print((df[['x', 'y', 'z']] == 0).sum())

df.info()
X = df.drop('price', axis=1)
y = df['price']

num_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
cat_columns = ['cut', 'color', 'clarity']

preprocessor = ColumnTransformer([('num', StandardScaler(), num_columns), ('cat', OneHotEncoder(drop='first'), cat_columns)])


# 3. Eğitim / Test Ayrımı ve Özellik Ölçekleme

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Eğitim seti boyutu:', X_train.shape)
print('Test seti boyutu  :', X_test.shape)

alphas = np.logspace(-4, 4, 100)


# 4. Ridge Regresyon (L2 Regularizasyon)

ridge_pipeline = Pipeline(steps=[('Preprocessor', preprocessor), ('Ridge', Ridge(alpha=1))])

ridge_pipeline.fit(X_train, y_train)
y_pred = ridge_pipeline.predict(X_test)
print('Ridge R2:', r2_score(y_test, y_pred))
print('Ridge MSE:', mean_squared_error(y_test, y_pred))
print('Ridge MAE:', mean_absolute_error(y_test, y_pred))

corr_matrix = df[num_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
save_figure('diamonds_correlation.png')

alphas = np.logspace(-3, 3, 10)

ridge_pipeline = Pipeline(steps=[('Preprocessor', preprocessor), ('Ridge', RidgeCV(alphas=alphas, cv=5))])

ridge_pipeline.fit(X_train, y_train)
y_pred = ridge_pipeline.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_pred)
ridge_mae = mean_absolute_error(y_test, y_pred)
ridge_r2 = r2_score(y_test, y_pred)
print('Ridge R2:', r2_score(y_test, y_pred))
print('Ridge MSE:', mean_squared_error(y_test, y_pred))
print('Ridge MAE:', mean_absolute_error(y_test, y_pred))
best_alpha = ridge_pipeline.named_steps['Ridge'].alpha_
print('Best alpha:', best_alpha)


# 5. Lasso Regresyon (L1 Regularizasyon)

lasso_pipeline = Pipeline(steps=[('Preprocessor', preprocessor), ('Lasso', Lasso(alpha=1))])
lasso_pipeline.fit(X_train, y_train)
y_pred_lasso = lasso_pipeline.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
print(f'Lasso -> MSE: {lasso_mse:.2f} | MAE: {lasso_mae:.2f}')
lasso_r2 = r2_score(y_test, y_pred_lasso)
print(f'Lasso -> R2: {lasso_r2:.4f}')

from sklearn.linear_model import LassoCV
lassocv_pipeline = Pipeline(steps=[('Preprocessor', preprocessor), ('Lasso', LassoCV(alphas=alphas, cv=5))])

lassocv_pipeline.fit(X_train, y_train)
best_alpha_lasso = lassocv_pipeline.named_steps['Lasso'].alpha_

print('Best alpha:', best_alpha_lasso)


# 6. Elastic Net Regresyon (L1 + L2 Kombinasyonu)

elasticnet_pipeline = Pipeline(steps=[('Preprocessor', preprocessor), ('ElasticNet', ElasticNet(alpha=1, l1_ratio=0.5, max_iter=1000))])

elasticnet_pipeline.fit(X_train, y_train)

y_pred_elasticnet = elasticnet_pipeline.predict(X_test)
elastic_mse = mean_squared_error(y_test, y_pred_elasticnet)
elastic_mae = mean_absolute_error(y_test, y_pred_elasticnet)
print(f'ElasticNet -> MSE: {elastic_mse:.2f} | MAE: {elastic_mae:.2f}')
elastic_r2 = r2_score(y_test, y_pred_elasticnet)
print(f'ElasticNet -> R2: {elastic_r2:.4f}')


# 7. Model Karşılaştırması

sonuclar = pd.DataFrame({'Model': ['Ridge', 'Lasso', 'ElasticNet'], 'MSE': [ridge_mse, lasso_mse, elastic_mse], 'MAE': [ridge_mae, lasso_mae, elastic_mae], 'R2': [ridge_r2, lasso_r2, elastic_r2]})
sonuclar = sonuclar.sort_values('R2', ascending=False)
sonuclar

X

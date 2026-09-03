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


# 1. ZAMAN SERİSİ KAVRAMLARI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
print('Kütüphaneler yüklendi')


# 2. VERİ HAZIRLAMA

df = pd.read_csv(DATA_DIR / 'monthly_passengers.csv', parse_dates=['Date']).set_index('Date')
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]
print(f'Veri seti boyutu: {len(df)}')
print(f'Eğitim: {len(train_data)}, Test: {len(test_data)}')

df


# 3. DURAĞANLIK TESTLERİ

adf_result = adfuller(train_data['Passengers'].values, autolag='AIC')
print('ADF Testi Sonuçları:')
print(f'Test İstatistiği: {adf_result[0]:.6f}')
print(f'P-Değeri: {adf_result[1]:.6f}')
if adf_result[1] <= 0.05:
    print('Sonuç: Seri DURAĞAN')
else:
    print('Sonuç: Seri DURAĞAN DEĞİL - Fark alma gerekli')


# 4. ARIMA MODELİ

from pmdarima import auto_arima
print('Optimal ARIMA parametreleri bulunuyor...')
auto_model = auto_arima(train_data['Passengers'], seasonal=False, max_p=5, max_q=5, max_d=2, trace=False, suppress_warnings=True)
print(f'Optimal ARIMA: ARIMA{auto_model.order}')

arima_model = ARIMA(train_data['Passengers'], order=auto_model.order)
arima_results = arima_model.fit()
print(f'AIC: {arima_results.aic:.2f}')
print(f'BIC: {arima_results.bic:.2f}')

arima_predictions = arima_results.get_forecast(steps=len(test_data))
arima_pred_values = arima_predictions.predicted_mean
arima_rmse = np.sqrt(mean_squared_error(test_data['Passengers'].values, arima_pred_values.values))
arima_mae = mean_absolute_error(test_data['Passengers'].values, arima_pred_values.values)
print(f'ARIMA RMSE: {arima_rmse:.2f}')
print(f'ARIMA MAE: {arima_mae:.2f}')


# 5. SARIMA MODELİ

print('Optimal SARIMA parametreleri bulunuyor...')
sarima_auto = auto_arima(train_data['Passengers'], seasonal=True, m=12, max_p=5, max_q=5, max_d=2, D=0, max_P=2, max_Q=2, trace=False, suppress_warnings=True)
print(f'Optimal SARIMA: SARIMA{sarima_auto.order}{sarima_auto.seasonal_order}')

sarima_model = SARIMAX(train_data['Passengers'], order=sarima_auto.order, seasonal_order=sarima_auto.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
sarima_results = sarima_model.fit(disp=False)
print(f'AIC: {sarima_results.aic:.2f}')

sarima_predictions = sarima_results.get_forecast(steps=len(test_data))
sarima_pred_values = sarima_predictions.predicted_mean
sarima_rmse = np.sqrt(mean_squared_error(test_data['Passengers'].values, sarima_pred_values.values))
sarima_mae = mean_absolute_error(test_data['Passengers'].values, sarima_pred_values.values)
print(f'SARIMA RMSE: {sarima_rmse:.2f}')
print(f'SARIMA MAE: {sarima_mae:.2f}')


# 6. KARŞILAŞTIRMA

results = pd.DataFrame({'Model': ['ARIMA', 'SARIMA'], 'RMSE': [arima_rmse, sarima_rmse], 'MAE': [arima_mae, sarima_mae]})
print('\nModel Karşılaştırması:')
print(results.to_string(index=False))
best_model = results.loc[results['RMSE'].idxmin()]
print(f"\nEn iyi model: {best_model['Model']} (RMSE: {best_model['RMSE']:.2f})")


# 7. TAHMİNLERİN GÖRSELLEŞTİRİLMESİ

plt.figure(figsize=(14, 6))
plt.plot(train_data.index, train_data['Passengers'], label='Eğitim', marker='o', linewidth=2)
plt.plot(test_data.index, test_data['Passengers'], label='Test', marker='s', linewidth=2)
plt.plot(test_data.index, arima_pred_values, label='ARIMA', marker='^', linewidth=2, alpha=0.7)
plt.plot(test_data.index, sarima_pred_values, label='SARIMA', marker='v', linewidth=2, alpha=0.7)
plt.axvline(x=train_data.index[-1], color='red', linestyle='--', alpha=0.5)
plt.title('ARIMA vs SARIMA Tahminleri')
plt.xlabel('Tarih')
plt.ylabel('Yolcu Sayısı')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_figure('time_series_components.png')


# 8. GELECEK TAHMİNLERİ

full_model = SARIMAX(df['Passengers'], order=sarima_auto.order, seasonal_order=sarima_auto.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
full_results = full_model.fit(disp=False)
forecast = full_results.get_forecast(steps=12)
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()
print('Gelecek 12 ay tahminleri:')
for i, val in enumerate(forecast_values, 1):
    print(f'Ay {i}: {val:.0f}')

last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Passengers'], label='Tarihsel Veri', linewidth=2)
plt.plot(future_dates, forecast_values, label='Tahminler', marker='o', linewidth=2, color='red')
plt.fill_between(future_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.2, color='red')
plt.axvline(x=df.index[-1], color='black', linestyle=':', alpha=0.5)
plt.title('Gelecek 12 Ay Tahminleri')
plt.xlabel('Tarih')
plt.ylabel('Yolcu Sayısı')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_figure('forecast_comparison.png')

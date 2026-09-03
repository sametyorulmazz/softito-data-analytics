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


# 1. Kütüphaneler, tekrar üretilebilirlik ve ortam bilgisi

import os
import sys
import platform
import warnings
import random
from datetime import date
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 50)
plt.rcParams['figure.figsize'] = (13, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
print(f'Python    : {sys.version.split()[0]}')
print(f'Platform  : {platform.platform()}')
print(f'pandas    : {pd.__version__}')
print(f'numpy     : {np.__version__}')
print(f'lightgbm  : {lgb.__version__}')
print(f'Çalıştırma tarihi: {date.today()}')


# 2. Yapılandırma (tek noktadan kontrol)

TICKER = 'AAPL'
START_DATE = '2015-01-01'
END_DATE = '2017-02-16'
INTERVAL = '1d'
HORIZON = 1
TEST_SIZE = 88
VAL_SIZE = 88
FUTURE_DAYS = 15
KATEGORIK = ['haftanin_gunu', 'ay']
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Hedef: {TICKER} | {START_DATE} → {END_DATE or 'bugün'} | ufuk: t+{HORIZON} gün")


# 3. Yerel fiyat verisini yükleme

def veri_indir(ticker: str, start: str, end=None, interval: str='1d') -> pd.DataFrame:
    path = DATA_DIR / 'aapl_prices.csv'
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.rename(columns={
        'AAPL.Open': 'Open', 'AAPL.High': 'High', 'AAPL.Low': 'Low',
        'AAPL.Close': 'Close', 'AAPL.Volume': 'Volume',
    }).set_index('Date')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
raw = veri_indir(TICKER, START_DATE, END_DATE, INTERVAL)
print(f'Satır sayısı : {len(raw)}')
print(f'Tarih aralığı: {raw.index.min().date()} → {raw.index.max().date()}')
print(f'Sütunlar     : {list(raw.columns)}')
raw.tail()


# Veri kalite kontrolü

df = raw.copy()
print('--- Eksik değer sayısı ---')
print(df.isna().sum())
print('\n--- Betimsel istatistik ---')
print(df[['Close', 'Volume']].describe().round(2))
assert (df['Close'] > 0).all(), 'Sıfır/negatif kapanış fiyatı var!'
bosluk = df.index.to_series().diff().dt.days
print(f'\nEn uzun tarih boşluğu: {int(bosluk.max())} gün ({df.index[int(bosluk.argmax())].date()})')
print(f"Sıfır hacimli gün sayısı: {int((df['Volume'] == 0).sum())}")


# 4. Keşifsel veri analizi (EDA)

df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
fig, axes = plt.subplots(3, 1, figsize=(13, 11))
axes[0].plot(df.index, df['Close'], lw=1.1, color='#1f4e79')
axes[0].set_title(f'{TICKER} — Düzeltilmiş Kapanış Fiyatı')
axes[0].set_ylabel('Fiyat')
axes[1].bar(df.index, df['Volume'], width=1.0, color='#7f8c8d')
axes[1].set_title('İşlem Hacmi')
axes[1].set_ylabel('Adet')
r = df['log_ret'].dropna()
axes[2].hist(r, bins=100, density=True, alpha=0.7, color='#27ae60')
x = np.linspace(r.min(), r.max(), 300)
axes[2].plot(x, 1 / (r.std() * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - r.mean()) / r.std()) ** 2), 'k--', lw=1.5, label='Normal dağılım')
axes[2].set_title(f'Günlük Log Getiri Dağılımı — Çarpıklık: {r.skew():.2f}, Basıklık: {r.kurt():.2f}')
axes[2].legend()
plt.tight_layout()
save_figure('aapl_eda.png')
print(f'Yıllık ortalama getiri (log) : {r.mean() * 252:.2%}')
print(f'Yıllık volatilite           : {r.std() * np.sqrt(252):.2%}')


# 5. Özellik mühendisliği

def rsi_hesapla(seri: pd.Series, periyot: int=14) -> pd.Series:
    delta = seri.diff()
    kazanc = delta.clip(lower=0)
    kayip = (-delta).clip(lower=0)
    ort_kazanc = kazanc.ewm(alpha=1 / periyot, adjust=False).mean()
    ort_kayip = kayip.ewm(alpha=1 / periyot, adjust=False).mean()
    rs = ort_kazanc / ort_kayip.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd_hesapla(seri: pd.Series, hizli: int=12, yavas: int=26, sinyal: int=9):
    ema_hizli = seri.ewm(span=hizli, adjust=False).mean()
    ema_yavas = seri.ewm(span=yavas, adjust=False).mean()
    macd = ema_hizli - ema_yavas
    sinyal_cizgi = macd.ewm(span=sinyal, adjust=False).mean()
    return (macd, sinyal_cizgi, macd - sinyal_cizgi)


# Özellik üretim fonksiyonu

def ozellik_uret(data: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    kapanis = data['Close'].astype(float)
    hacim = data['Volume'].astype(float)
    ret = np.log(kapanis / kapanis.shift(1))
    out['ret_1'] = ret
    for k in range(1, 6):
        out[f'ret_lag_{k}'] = ret.shift(k)
    for n in (5, 10, 21, 63):
        out[f'mom_{n}'] = np.log(kapanis / kapanis.shift(n))
    for n in (5, 10, 20, 50):
        sma = kapanis.rolling(n).mean()
        out[f'close_sma_{n}'] = kapanis / sma - 1
    out['sma_5_20'] = kapanis.rolling(5).mean() / kapanis.rolling(20).mean() - 1
    for n in (5, 10, 21):
        out[f'vol_{n}'] = ret.rolling(n).std()
    out['vol_orani'] = out['vol_5'] / out['vol_21'].replace(0, np.nan)
    out['rsi_14'] = rsi_hesapla(kapanis, 14) / 100
    _, _, macd_hist = macd_hesapla(kapanis)
    out['macd_hist'] = macd_hist / kapanis
    orta = kapanis.rolling(20).mean()
    std = kapanis.rolling(20).std()
    out['bb_pct'] = (kapanis - (orta - 2 * std)) / (4 * std).replace(0, np.nan)
    hacim_g = hacim.replace(0, np.nan)
    out['hacim_orani'] = hacim_g / hacim_g.rolling(20).mean()
    out['hacim_degisim'] = np.log(hacim_g / hacim_g.shift(1))
    out['haftanin_gunu'] = pd.Categorical(data.index.dayofweek, categories=range(7))
    out['ay'] = pd.Categorical(data.index.month, categories=range(1, 13))
    sayisal = out.select_dtypes(include=[np.number]).columns
    out[sayisal] = out[sayisal].replace([np.inf, -np.inf], np.nan)
    return out
ozellikler = ozellik_uret(df)
print(f'Üretilen özellik sayısı: {ozellikler.shape[1]}')
print(f"Kategorik sütunlar     : {list(ozellikler.select_dtypes('category').columns)}")
ozellikler.tail(3)


# 6. Hedef değişken: neden fiyat değil, **getiri**?

hedef = np.log(df['Close'].shift(-HORIZON) / df['Close'])
veri = ozellikler.copy()
veri['y'] = hedef
veri['Close'] = df['Close']
veri = veri.dropna()
FEATURES = [c for c in veri.columns if c not in ('y', 'Close')]
X = veri[FEATURES]
y = veri['y']
fiyat = veri['Close']
print(f'Örneklem     : {len(veri)} satır × {len(FEATURES)} özellik')
print(f'Tarih aralığı: {veri.index.min().date()} → {veri.index.max().date()}')
print(f'Hedef std    : {y.std():.5f}  |  ortalama: {y.mean():.6f}')


# 7. Zaman serisi bölmesi (kronolojik)

n = len(veri)
test_bas = n - TEST_SIZE
val_bas = test_bas - VAL_SIZE
assert val_bas > 250, "Veri çok kısa; START_DATE'i geriye çekin veya TEST_SIZE'ı küçültün."
X_egt, y_egt = (X.iloc[:val_bas], y.iloc[:val_bas])
X_val, y_val = (X.iloc[val_bas:test_bas], y.iloc[val_bas:test_bas])
X_test, y_test = (X.iloc[test_bas:], y.iloc[test_bas:])
fiyat_test = fiyat.iloc[test_bas:]
for ad, blok in [('Eğitim', X_egt), ('Doğrulama', X_val), ('Test', X_test)]:
    print(f'{ad:<10}: {len(blok):>5} satır | {blok.index.min().date()} → {blok.index.max().date()}')


# 8. Baseline (referans) model — en kritik adım

def metrikler(y_gercek, y_tahmin, fiyat_t, etiket=''):
    y_gercek = np.asarray(y_gercek, dtype=float)
    y_tahmin = np.asarray(y_tahmin, dtype=float)
    p_t = np.asarray(fiyat_t, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_gercek, y_tahmin)))
    mae = float(mean_absolute_error(y_gercek, y_tahmin))
    r2 = float(r2_score(y_gercek, y_tahmin))
    gercek_fiyat = p_t * np.exp(y_gercek)
    tahmini_fiyat = p_t * np.exp(y_tahmin)
    mape = float(np.mean(np.abs(gercek_fiyat - tahmini_fiyat) / gercek_fiyat) * 100)
    maske = y_tahmin != 0
    yon = float((np.sign(y_gercek[maske]) == np.sign(y_tahmin[maske])).mean() * 100) if maske.any() else np.nan
    return {'Model': etiket, 'RMSE(getiri)': rmse, 'MAE(getiri)': mae, 'R2': r2, 'MAPE(fiyat)%': mape, 'Yön doğruluğu%': yon}
naif_tahmin = np.zeros(len(y_test))
m_naif = metrikler(y_test, naif_tahmin, fiyat_test, 'Naif (yarın=bugün)')
ort_tahmin = np.full(len(y_test), y_egt.mean())
m_ort = metrikler(y_test, ort_tahmin, fiyat_test, 'Sabit drift (eğitim ort.)')
pd.DataFrame([m_naif, m_ort]).round(5)


# 9. LightGBM modelinin eğitilmesi

model = LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=7, max_depth=3, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1, min_split_gain=0.0, max_bin=255, objective='regression', random_state=SEED, n_jobs=-1, verbosity=-1)
gecmis = {}
model.fit(X_egt, y_egt, eval_set=[(X_egt, y_egt), (X_val, y_val)], eval_names=['egitim', 'dogrulama'], eval_metric='rmse', categorical_feature=KATEGORIK, callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0), lgb.record_evaluation(gecmis)])
print(f'En iyi tur (ağaç sayısı): {model.best_iteration_}')
print(f"En iyi doğrulama RMSE   : {model.best_score_['dogrulama']['rmse']:.6f}")
plt.figure(figsize=(11, 4))
plt.plot(gecmis['egitim']['rmse'], label='Eğitim RMSE', lw=1.4)
plt.plot(gecmis['dogrulama']['rmse'], label='Doğrulama RMSE', lw=1.4)
plt.axvline(model.best_iteration_, color='red', ls='--', label=f'En iyi tur = {model.best_iteration_}')
plt.xlabel('Boosting turu')
plt.ylabel('RMSE')
plt.title('LightGBM Öğrenme Eğrisi')
plt.legend()
plt.tight_layout()
save_figure('learning_curve.png')


# 10. Test seti üzerinde değerlendirme

tahmin_test = model.predict(X_test)
m_lgb = metrikler(y_test, tahmin_test, fiyat_test, 'LightGBM')
karsilastirma = pd.DataFrame([m_naif, m_ort, m_lgb]).round(5)
print(karsilastirma.to_string(index=False))
iyilesme = (m_naif['RMSE(getiri)'] - m_lgb['RMSE(getiri)']) / m_naif['RMSE(getiri)'] * 100
print(f'\nRMSE iyileşmesi (naife göre): {iyilesme:+.2f}%')
print('YORUM:', 'Model naif tahmini yeniyor.' if iyilesme > 0 else 'Model naif tahmini YENEMİYOR — beklenen bir sonuçtur, dürüstçe raporlayın.')

gercek_p = fiyat_test.values * np.exp(y_test.values)
tahmini_p = fiyat_test.values * np.exp(tahmin_test)
tarih_p = fiyat_test.index
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
ax[0, 0].plot(tarih_p, gercek_p, label='Gerçek', lw=1.4, color='#1f4e79')
ax[0, 0].plot(tarih_p, tahmini_p, label='LightGBM', lw=1.2, ls='--', color='#27ae60')
ax[0, 0].set_title(f'Test Dönemi — Gerçek vs Tahmin (t+{HORIZON})')
ax[0, 0].legend()
ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax[0, 1].scatter(y_test, tahmin_test, s=12, alpha=0.55, color='#2c7873')
lim = float(np.abs(y_test).max())
ax[0, 1].plot([-lim, lim], [-lim, lim], 'k--', lw=1)
ax[0, 1].axhline(0, color='gray', lw=0.8)
ax[0, 1].axvline(0, color='gray', lw=0.8)
ax[0, 1].set_xlabel('Gerçek getiri')
ax[0, 1].set_ylabel('Tahmin edilen getiri')
ax[0, 1].set_title('Tahmin dağılımı (ölçek farkına dikkat)')
artiklar = y_test.values - tahmin_test
ax[1, 0].hist(artiklar, bins=50, color='#7f8c8d', alpha=0.85)
ax[1, 0].axvline(0, color='red', ls='--')
ax[1, 0].set_title(f'Artıklar — ortalama: {artiklar.mean():.5f}, std: {artiklar.std():.5f}')
ax[1, 1].plot(tarih_p, artiklar, lw=0.9, color='#8e44ad')
ax[1, 1].axhline(0, color='red', ls='--')
ax[1, 1].set_title('Artıkların zaman serisi (hata varyansı sabit mi?)')
ax[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
save_figure('test_diagnostics.png')


# 11. Hiperparametre optimizasyonu — `TimeSeriesSplit` ile

param_uzayi = {'n_estimators': [300, 500, 800], 'learning_rate': [0.005, 0.01, 0.03], 'num_leaves': [3, 7, 15, 31], 'max_depth': [2, 3, 4, -1], 'min_child_samples': [20, 30, 50, 100], 'subsample': [0.7, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0], 'reg_lambda': [0.5, 1.0, 5.0], 'reg_alpha': [0.0, 0.1, 1.0]}
tscv = TimeSeriesSplit(n_splits=5)
arama = RandomizedSearchCV(estimator=LGBMRegressor(objective='regression', random_state=SEED, n_jobs=1, verbosity=-1, subsample_freq=1), param_distributions=param_uzayi, n_iter=25, scoring='neg_root_mean_squared_error', cv=tscv, random_state=SEED, n_jobs=-1, verbose=0)
X_ara = pd.concat([X_egt, X_val])
y_ara = pd.concat([y_egt, y_val])
arama.fit(X_ara, y_ara)
print('En iyi CV RMSE :', round(-arama.best_score_, 6))
print('En iyi parametreler:')
for k, v in sorted(arama.best_params_.items()):
    print(f'   {k:<20} = {v}')
en_iyi = arama.best_estimator_
tahmin_opt = en_iyi.predict(X_test)
m_opt = metrikler(y_test, tahmin_opt, fiyat_test, 'LightGBM (optimize)')
pd.DataFrame([m_naif, m_lgb, m_opt]).round(5)


# 12. Walk-forward (kayan pencere) doğrulama

def walk_forward(X, y, fiyat, baslangic, adim=21, params=None):
    params = params or dict(n_estimators=500, learning_rate=0.01, num_leaves=7, max_depth=3, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1, objective='regression', random_state=SEED, n_jobs=-1, verbosity=-1)
    tahminler, gercekler, fiyatlar, tarihler = ([], [], [], [])
    for bas in range(baslangic, len(X), adim):
        son = min(bas + adim, len(X))
        m = LGBMRegressor(**params)
        m.fit(X.iloc[:bas], y.iloc[:bas], categorical_feature=KATEGORIK)
        tahminler.append(m.predict(X.iloc[bas:son]))
        gercekler.append(y.iloc[bas:son].values)
        fiyatlar.append(fiyat.iloc[bas:son].values)
        tarihler.append(X.index[bas:son])
    return (np.concatenate(tahminler), np.concatenate(gercekler), np.concatenate(fiyatlar), np.concatenate(tarihler))
wf_tahmin, wf_gercek, wf_fiyat, wf_tarih = walk_forward(X, y, fiyat, baslangic=val_bas, adim=21)
m_wf = metrikler(wf_gercek, wf_tahmin, wf_fiyat, 'LightGBM (walk-forward)')
m_wf_naif = metrikler(wf_gercek, np.zeros_like(wf_gercek), wf_fiyat, 'Naif (walk-forward)')
print(pd.DataFrame([m_wf_naif, m_wf]).round(5).to_string(index=False))
SIGMA = float(np.std(wf_gercek - wf_tahmin))
print(f'\nWalk-forward artık std (SIGMA): {SIGMA:.5f}  (günlük log getiri biriminde)')


# 13. Özellik önemleri

booster = model.booster_
adlar = booster.feature_name()
onem = pd.DataFrame({'gain': booster.feature_importance(importance_type='gain'), 'split': booster.feature_importance(importance_type='split')}, index=adlar)
onem['gain_%'] = onem['gain'] / onem['gain'].sum() * 100
onem['split_%'] = onem['split'] / onem['split'].sum() * 100
onem = onem.sort_values('gain_%', ascending=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
onem['gain_%'].head(20)[::-1].plot(kind='barh', ax=ax[0], color='#27ae60')
ax[0].set_title('Önem — gain (%)')
ax[0].set_xlabel('Göreli önem')
onem['split_%'].head(20)[::-1].plot(kind='barh', ax=ax[1], color='#95a5a6')
ax[1].set_title('Önem — split (%)')
ax[1].set_xlabel('Bölme payı')
plt.tight_layout()
save_figure('feature_importance.png')
print(onem[['gain_%', 'split_%']].head(10).round(2).to_string())


# 14. Geleceğe yönelik özyinelemeli (recursive) tahmin

def gelecegi_tahmin_et(model, kaynak_df, feature_cols, n_gun, sigma=None, z=1.96):
    sim = kaynak_df[['Close', 'Volume']].copy().astype(float)
    varsayilan_hacim = float(sim['Volume'].tail(20).median())
    kayitlar = []
    for adim in range(1, n_gun + 1):
        f = ozellik_uret(sim)
        x_son = f[feature_cols].iloc[[-1]]
        sayisal = x_son.select_dtypes(include=[np.number]).columns
        x_son[sayisal] = x_son[sayisal].fillna(0.0)
        r_hat = float(model.predict(x_son)[0])
        son_fiyat = float(sim['Close'].iloc[-1])
        yeni_fiyat = son_fiyat * np.exp(r_hat)
        yeni_tarih = sim.index[-1] + pd.tseries.offsets.BDay(1)
        sim.loc[yeni_tarih] = {'Close': yeni_fiyat, 'Volume': varsayilan_hacim}
        if sigma is not None:
            genislik = z * sigma * np.sqrt(adim)
            alt = yeni_fiyat * np.exp(-genislik)
            ust = yeni_fiyat * np.exp(genislik)
        else:
            alt = ust = np.nan
        kayitlar.append({'tarih': yeni_tarih, 'tahmin_getiri': r_hat, 'tahmin_fiyat': yeni_fiyat, 'alt_%95': alt, 'ust_%95': ust})
    return pd.DataFrame(kayitlar).set_index('tarih')
nihai_params = dict(arama.best_params_)
nihai_params.update(objective='regression', random_state=SEED, n_jobs=-1, verbosity=-1, subsample_freq=1)
nihai_model = LGBMRegressor(**nihai_params)
nihai_model.fit(X, y, categorical_feature=KATEGORIK)
gelecek = gelecegi_tahmin_et(nihai_model, df, FEATURES, FUTURE_DAYS, sigma=SIGMA)
print(f"Son gerçek kapanış: {df['Close'].iloc[-1]:.2f}  ({df.index[-1].date()})\n")
print(gelecek.round(3).to_string())

gecmis_pencere = 120
gecmis_seri = df['Close'].iloc[-gecmis_pencere:]
plt.figure(figsize=(13, 6))
plt.plot(gecmis_seri.index, gecmis_seri.values, label='Gerçekleşen fiyat', lw=1.6, color='#1f4e79')
kopru_x = [gecmis_seri.index[-1]] + list(gelecek.index)
kopru_y = [gecmis_seri.values[-1]] + list(gelecek['tahmin_fiyat'])
plt.plot(kopru_x, kopru_y, label=f'LightGBM projeksiyonu ({FUTURE_DAYS} iş günü)', lw=1.8, ls='--', marker='o', ms=3.5, color='#27ae60')
plt.fill_between(gelecek.index, gelecek['alt_%95'], gelecek['ust_%95'], alpha=0.18, color='#27ae60', label='%95 belirsizlik bandı')
plt.axvline(gecmis_seri.index[-1], color='gray', ls=':', lw=1.2)
plt.title(f'{TICKER} — Geçmiş ve {FUTURE_DAYS} Günlük Projeksiyon')
plt.ylabel('Fiyat')
plt.legend(loc='upper left')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
save_figure('future_projection.png')
toplam = (gelecek['tahmin_fiyat'].iloc[-1] / df['Close'].iloc[-1] - 1) * 100
print(f'{FUTURE_DAYS} günlük kümülatif tahmini değişim: {toplam:+.2f}%')
print(f"%95 bant (son gün): {gelecek['alt_%95'].iloc[-1]:.2f} — {gelecek['ust_%95'].iloc[-1]:.2f}")
print('\nBandın genişliğine dikkat: nokta tahminden çok daha bilgilendiricidir.')


# 15. LightGBM vs XGBoost — aynı bölmede yarıştırma

try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_estimators=max(model.best_iteration_ or 0, 300), learning_rate=0.01, max_depth=3, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1, objective='reg:squarederror', random_state=SEED, n_jobs=-1)
    X_egt_x = X_egt.copy()
    X_test_x = X_test.copy()
    for c in KATEGORIK:
        X_egt_x[c] = X_egt_x[c].astype(int)
        X_test_x[c] = X_test_x[c].astype(int)
    xgb_model.fit(X_egt_x, y_egt, verbose=False)
    m_xgb = metrikler(y_test, xgb_model.predict(X_test_x), fiyat_test, 'XGBoost')
    print(pd.DataFrame([m_naif, m_lgb, m_opt, m_xgb]).round(5).to_string(index=False))
    fark = (m_xgb['RMSE(getiri)'] - m_lgb['RMSE(getiri)']) / m_xgb['RMSE(getiri)'] * 100
    print(f"\nLightGBM, XGBoost'a göre RMSE farkı: {fark:+.2f}% ({('LightGBM daha iyi' if fark > 0 else 'XGBoost daha iyi')})")
    print('Not: Bu fark genellikle istatistiksel olarak anlamsızdır; farklı SEED değerleriyle tekrarlayıp dağılıma bakın.')
except ImportError:
    print('xgboost kurulu değil — karşılaştırma atlandı.  %pip install -q xgboost')


# 16. Model kaydetme, sürüm bilgisi ve raporlama bloğu

model_yolu = os.path.join(OUTPUT_DIR, f"lgbm_{TICKER.replace('.', '_')}.txt")
nihai_model.booster_.save_model(model_yolu)
meta = {'ticker': TICKER, 'baslangic': START_DATE, 'bitis': str(df.index[-1].date()), 'ufuk_gun': HORIZON, 'ozellikler': FEATURES, 'kategorik': KATEGORIK, 'en_iyi_parametreler': arama.best_params_, 'lightgbm_surum': lgb.__version__, 'seed': SEED}
pd.Series(meta).to_json(os.path.join(OUTPUT_DIR, 'model_meta.json'), force_ascii=False)
print(f'Kaydedildi → {model_yolu}')
print('\n' + '=' * 78)
print('BULGULAR (rapor taslağı)')
print('=' * 78)
print(f"\n{TICKER} hissesinin {df.index[0].date()}–{df.index[-1].date()} dönemine ait günlük\nfiyat ve hacim verileri yerel Plotly veri setinden okunmuştur (n = {len(veri)}).\nTeknik göstergelerden {len(FEATURES)} özellik türetilmiş ({len(KATEGORIK)} tanesi kategorik\nolarak modellenmiştir), hedef değişken olarak t+{HORIZON} günlük logaritmik getiri\nkullanılmıştır. Veri kronolojik olarak eğitim (n={len(X_egt)}), doğrulama (n={len(X_val)})\nve test (n={len(X_test)}) alt kümelerine ayrılmıştır. Model olarak LightGBM\n(leaf-wise, histogram tabanlı gradyan artırma; num_leaves ve min_child_samples ile\ndüzenlileştirilmiş) kullanılmış, ağaç sayısı doğrulama kümesinde erken durdurma ile\nbelirlenmiştir (en iyi tur = {model.best_iteration_}).\n\nTest seti sonuçları:\n  Naif (rastgele yürüyüş) RMSE = {m_naif['RMSE(getiri)']:.5f}\n  LightGBM RMSE                = {m_lgb['RMSE(getiri)']:.5f}\n  LightGBM R²                  = {m_lgb['R2']:.4f}\n  LightGBM yön doğruluğu       = %{m_lgb['Yön doğruluğu%']:.1f}\n  Naife göre RMSE değişimi     = %{iyilesme:+.2f}\n\nWalk-forward doğrulama (n = {len(wf_gercek)}, aylık yeniden eğitim):\n  RMSE = {m_wf['RMSE(getiri)']:.5f} | R² = {m_wf['R2']:.4f} | Yön doğruluğu = %{m_wf['Yön doğruluğu%']:.1f}\n\nSonuç: Model, günlük getiri varyansının yalnızca sınırlı bir kısmını açıklamaktadır.\nBu bulgu, zayıf formda etkin piyasa hipoteziyle tutarlıdır ve tek başına bir\nalım-satım stratejisinin dayanağı olarak kullanılamaz.\n")

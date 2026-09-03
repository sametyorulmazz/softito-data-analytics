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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/student.csv')
print('Boyut:', df.shape)
print('Eksik değer toplamı:', int(df.isna().sum().sum()))
df[['productivity_score', 'addiction_level']].head()


# 1. Üretkenlik puanı: regresyon

features = [c for c in df.columns if c not in ['productivity_score', 'addiction_level']]
reg_df = df.dropna(subset=['productivity_score']).copy()
X_reg = reg_df[features]
y_reg = reg_df['productivity_score']
reg = Pipeline([('impute', SimpleImputer(strategy='median')), ('model', RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1))])
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
scores = cross_validate(reg, X_reg, y_reg, cv=rkf, scoring=['r2', 'neg_mean_absolute_error'])
pd.DataFrame({'R2': scores['test_r2'], 'MAE': -scores['test_neg_mean_absolute_error']}).agg(['mean', 'std']).round(3)


# 2. Bağımlılık seviyesi: sınıflandırma

cls_df = df.dropna(subset=['addiction_level']).copy()
X_cls = cls_df[features]
y_cls = cls_df['addiction_level']
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42)
clf = Pipeline([('impute', SimpleImputer(strategy='median')), ('model', RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(clf, X_train, y_train, cv=cv, scoring=['accuracy', 'f1_macro'])
pd.DataFrame(cv_scores).filter(like='test_').agg(['mean', 'std']).T.round(3)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred, digits=3))
perm = permutation_importance(clf, X_test, y_test, scoring='f1_macro', n_repeats=5, random_state=42, n_jobs=-1)
importance = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 4.5))
importance.sort_values().plot.barh(ax=ax, color='#4472C4')
ax.set_title('Bağımlılık Sınıflandırması: Permütasyon Önemi')
ax.set_xlabel('Macro-F1 düşüşü')
plt.tight_layout()
save_figure('student_feature_importance.png')

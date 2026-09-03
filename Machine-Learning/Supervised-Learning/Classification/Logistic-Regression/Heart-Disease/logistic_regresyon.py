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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

raw = pd.read_csv('data/heart.csv')
duplicate_count = int(raw.duplicated().sum())
df = raw.drop_duplicates().reset_index(drop=True)
print(f'Ham satır: {len(raw)} | Yinelenen: {duplicate_count} | Analiz satırı: {len(df)}')
df['target'].value_counts(normalize=True).rename('oran').round(3)


# Ön işleme ve değerlendirme

target = 'target'
categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric = [c for c in df.columns if c not in categorical + [target]]
X, y = (df.drop(columns=target), df[target])
preprocess = ColumnTransformer([('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric), ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)])
model = Pipeline([('prep', preprocess), ('model', LogisticRegression(max_iter=3000, class_weight='balanced'))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])
pd.DataFrame(cv_scores).filter(like='test_').agg(['mean', 'std']).T.round(3)

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred, digits=3))
print('Test ROC-AUC:', round(roc_auc_score(y_test, proba), 3))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axes[0], cmap='Blues', colorbar=False)
RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1], name='Lojistik regresyon')
axes[0].set_title('Karışıklık Matrisi')
axes[1].set_title('ROC Eğrisi')
axes[0].set_xlabel('Tahmin')
axes[0].set_ylabel('Gerçek')
axes[1].set_xlabel('Yanlış pozitif oranı')
axes[1].set_ylabel('Doğru pozitif oranı')
plt.tight_layout()
save_figure('heart_classification.png')

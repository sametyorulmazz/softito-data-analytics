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

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

df = pd.read_csv('data/india_job_market.csv', parse_dates=['Date_Posted'])
target = 'Experience_Level'
drop_cols = ['Job_ID', 'Company', 'Skills_Required', target]
X = df.drop(columns=drop_cols)
X['Posting_Month'] = df['Date_Posted'].dt.month
X = X.drop(columns='Date_Posted')
y = df[target]
print('Boyut:', df.shape)
y.value_counts(normalize=True).round(3)

numeric = X.select_dtypes(include='number').columns.tolist()
categorical = X.select_dtypes(exclude='number').columns.tolist()
prep = ColumnTransformer([('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric), ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)])
pipe = Pipeline([('prep', prep), ('svc', SVC(class_weight='balanced'))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
search = GridSearchCV(pipe, {'svc__C': [0.5, 1, 5], 'svc__kernel': ['linear', 'rbf'], 'svc__gamma': ['scale', 0.01]}, scoring='f1_macro', cv=3, n_jobs=-1)
search.fit(X_train, y_train)
print('En iyi parametreler:', search.best_params_)
print('CV macro-F1:', round(search.best_score_, 3))

pred = search.predict(X_test)
print(classification_report(y_test, pred, digits=3))
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap='Blues', xticks_rotation=30, colorbar=False)
ax.set_title('SVM Test Sonuçları')
plt.tight_layout()
save_figure('svm_confusion_matrix.png')

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes_health_indicators.csv"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# Veri hazırlama

df = pd.read_csv(DATA_PATH)
target = "Diabetes_012"
X = df.drop(columns=target)
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1_500,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ]
)


# Model eğitimi ve değerlendirme

pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
probability = pipeline.predict_proba(X_test)

accuracy = accuracy_score(y_test, prediction)
macro_f1 = f1_score(y_test, prediction, average="macro")
macro_auc = roc_auc_score(y_test, probability, multi_class="ovr", average="macro")

class_distribution = y.value_counts(normalize=True).sort_index().mul(100)
print(f"Veri boyutu: {df.shape}")
print("Sınıf dağılımı (%):")
print(class_distribution.round(2).to_string())
print(f"Accuracy: {accuracy:.3f}")
print(f"Macro-F1: {macro_f1:.3f}")
print(f"Macro ROC-AUC: {macro_auc:.3f}")


# Sonuç görselleri

labels = ["No diabetes", "Prediabetes", "Diabetes"]
matrix = confusion_matrix(y_test, prediction, normalize="true")
coefficients = pd.DataFrame(
    np.abs(pipeline.named_steps["model"].coef_),
    columns=X.columns,
).mean(axis=0).sort_values(ascending=False).head(12)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[0],
)
axes[0].set(title="Normalize Karışıklık Matrisi", xlabel="Tahmin", ylabel="Gerçek")

coefficients.sort_values().plot.barh(ax=axes[1], color="#E45756")
axes[1].set(title="Ortalama Mutlak Katsayılar", xlabel="|katsayı|")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "diabetes_classification_results.png", dpi=160, bbox_inches="tight")
plt.close()

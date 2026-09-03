from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telco_customer_churn.csv"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# Veri hazırlama

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer(
    [
        ("numeric", SimpleImputer(strategy="median"), numeric_columns),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_columns,
        ),
    ]
)

model = RandomForestClassifier(
    n_estimators=350,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)
probability = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, prediction)
macro_f1 = f1_score(y_test, prediction, average="macro")
roc_auc = roc_auc_score(y_test, probability)

print(f"Veri boyutu: {df.shape}")
print(f"Churn oranı: %{y.mean() * 100:.2f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Macro-F1: {macro_f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")


# Karışıklık matrisi ve özellik önemi

matrix = confusion_matrix(y_test, prediction)
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importance = pd.Series(
    pipeline.named_steps["model"].feature_importances_, index=feature_names
).sort_values(ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Kalır", "Ayrılır"],
    yticklabels=["Kalır", "Ayrılır"],
    ax=axes[0],
)
axes[0].set(title="Karışıklık Matrisi", xlabel="Tahmin", ylabel="Gerçek")

importance.sort_values().plot.barh(ax=axes[1], color="#4C78A8")
axes[1].set(title="En Önemli 15 Özellik", xlabel="Özellik önemi")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "telco_churn_results.png", dpi=160, bbox_inches="tight")
plt.close()

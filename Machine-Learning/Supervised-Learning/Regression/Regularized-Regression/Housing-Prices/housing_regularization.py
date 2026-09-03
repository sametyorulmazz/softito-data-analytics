from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "housing.csv"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# Veri hazırlama

df = pd.read_csv(DATA_PATH)
X = df.drop(columns="price")
y = np.log1p(df["price"])

numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_columns,
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            categorical_columns,
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


# Model seçimi

models = {
    "Ridge": (Ridge(), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
    "Lasso": (
        Lasso(max_iter=20_000),
        {"model__alpha": [0.0001, 0.001, 0.01, 0.1]},
    ),
    "Elastic Net": (
        ElasticNet(max_iter=20_000),
        {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
    ),
}

results = []
predictions = {}
fitted_models = {}

for name, (estimator, parameters) in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    search = GridSearchCV(pipeline, parameters, cv=5, scoring="neg_mean_absolute_error")
    search.fit(X_train, y_train)

    prediction = np.expm1(search.predict(X_test))
    actual = np.expm1(y_test)
    results.append(
        {
            "model": name,
            "mae": mean_absolute_error(actual, prediction),
            "rmse": mean_squared_error(actual, prediction) ** 0.5,
            "r2": r2_score(actual, prediction),
            "parameters": search.best_params_,
        }
    )
    predictions[name] = prediction
    fitted_models[name] = search.best_estimator_

results_df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
best_name = results_df.loc[0, "model"]
best_prediction = predictions[best_name]
actual = np.expm1(y_test).to_numpy()

print(f"Veri boyutu: {df.shape}")
print(results_df[["model", "mae", "rmse", "r2"]].round(3).to_string(index=False))
print(f"En iyi model: {best_name}")


# Sonuç görselleri

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(results_df["model"], results_df["mae"], color="#4C78A8")
axes[0].set(title="Model MAE Karşılaştırması", ylabel="MAE")
axes[0].tick_params(axis="x", rotation=15)

axes[1].scatter(actual, best_prediction, alpha=0.65, color="#F58518")
limits = [min(actual.min(), best_prediction.min()), max(actual.max(), best_prediction.max())]
axes[1].plot(limits, limits, "k--", linewidth=1)
axes[1].set(title=f"Gerçek ve Tahmin — {best_name}", xlabel="Gerçek fiyat", ylabel="Tahmin")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "housing_model_results.png", dpi=160, bbox_inches="tight")
plt.close()

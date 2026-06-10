# --------------------------------------------------------------------------------
# # Housing Price Regression Analysis
# 
# This notebook loads a housing price dataset, preprocesses it, trains several regression models, and compares their performance. It also performs a deeper residual analysis to identify when the best model is too optimistic or too pessimistic.
# --------------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data_paths = [
    Path("data/housing_price.csv"),
    Path("housing_price.csv"),
    Path("housing-price.csv"),
    Path("data/housing.csv"),
    Path("housing.csv"),
]
csv_path = next((p for p in data_paths if p.exists()), None)

if csv_path is not None:
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path}")
else:
    print("No local Kaggle housing CSV found. Falling back to sklearn California housing dataset.")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df["Price"] = df["MedHouseVal"] * 100000  # approximate dollar equivalent
    df = df.drop(columns=["MedHouseVal"])
    print("Using fallback California housing dataset with target column Price.")

print("Dataset shape:", df.shape)
display(df.head())
print("\nMissing values by column:")
display(df.isna().sum().sort_values(ascending=False).head(20))

possible_targets = ["Price", "price", "SalePrice", "saleprice", "Sale Price", "sale price", "HousePrice", "houseprice"]
target_col = next((col for col in df.columns if col in possible_targets), None)
if target_col is None:
    numeric_targets = [col for col in df.select_dtypes(include=["number"]).columns if "price" in col.lower() or "sale" in col.lower() or "value" in col.lower()]
    target_col = numeric_targets[0] if numeric_targets else df.columns[-1]  # fallback to last numeric column
print(f"Target column: {target_col}")
y = df[target_col]
X = df.drop(columns=[target_col])

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Lasso Regression": Lasso(alpha=0.1, random_state=42),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.2),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100, ), max_iter=500, random_state=42),
}

def evaluate_regression(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        "model": pipeline,
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "Predictions": y_pred,
    }

results = {}
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    results[name] = evaluate_regression(model, X_train, X_test, y_train, y_test)
print("Done.")

results_summary = pd.DataFrame([
    {
        "Model": name,
        "RMSE": res["RMSE"],
        "MAE": res["MAE"],
        "R2": res["R2"],
    }
    for name, res in results.items()
])
results_summary = results_summary.sort_values("RMSE")
display(results_summary)
best_model_name = results_summary.iloc[0]["Model"]
print(f"Best model by RMSE: {best_model_name}")

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(x="RMSE", y="Model", data=results_summary, ax=axes[0], palette="Blues_d")
axes[0].set_title("RMSE by Model")
sns.barplot(x="MAE", y="Model", data=results_summary, ax=axes[1], palette="Greens_d")
axes[1].set_title("MAE by Model")
sns.barplot(x="R2", y="Model", data=results_summary, ax=axes[2], palette="Oranges_d")
axes[2].set_title("R² by Model")
plt.tight_layout()
plt.show()

best_result = results[best_model_name]
best_pipeline = best_result["model"]
y_pred = best_result["Predictions"]
residuals = y_test - y_pred
analysis_df = X_test.copy()
analysis_df["Actual"] = y_test.values
analysis_df["Predicted"] = y_pred
analysis_df["Residual"] = residuals

print(f"Best model selected for residual analysis: {best_model_name}")
display(analysis_df[["Actual", "Predicted", "Residual"]].head())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x="Predicted", y="Actual", data=analysis_df, ax=axes[0])
axes[0].plot([analysis_df["Predicted"].min(), analysis_df["Predicted"].max()], [analysis_df["Predicted"].min(), analysis_df["Predicted"].max()], linestyle="--", color="gray")
axes[0].set_title("Actual vs Predicted")
axes[0].set_xlabel("Predicted price")
axes[0].set_ylabel("Actual price")
sns.histplot(analysis_df["Residual"], kde=True, ax=axes[1], color="firebrick")
axes[1].set_title("Residual distribution")
axes[1].set_xlabel("Actual - Predicted")
plt.tight_layout()
plt.show()

# Identify optimistic and pessimistic model errors
optimistic = analysis_df[analysis_df["Residual"] < 0].copy()
pessimistic = analysis_df[analysis_df["Residual"] > 0].copy()

optimistic = optimistic.assign(ErrorMagnitude=(-optimistic["Residual"]))
pessimistic = pessimistic.assign(ErrorMagnitude=pessimistic["Residual"].abs())

display(optimistic.sort_values("ErrorMagnitude", ascending=False).head(5)[["Predicted", "Actual", "Residual", "ErrorMagnitude"]])
display(pessimistic.sort_values("Residual", ascending=False).head(5)[["Predicted", "Actual", "Residual"]])

print("Top optimistic errors are cases where the model predicted too high.")
print("Top pessimistic errors are cases where the model predicted too low.")


import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier


# =============================
# 1. Load Dataset
# =============================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["customerID"])

# Separate features
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify numeric & categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", "passthrough", cat_cols)
    ]
)

# Convert categorical using get_dummies
X = pd.get_dummies(X, drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =============================
# 2. XGBoost Model
# =============================

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train) / sum(y_train)),  # handle imbalance
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# =============================
# 3. Evaluation
# =============================

preds = model.predict(X_val)
probs = model.predict_proba(X_val)[:, 1]

print("===== XGBoost Model =====")
print("Accuracy:", accuracy_score(y_val, preds))
print("F1 Score:", f1_score(y_val, preds))
print("ROC-AUC:", roc_auc_score(y_val, probs))

# =============================
# 4. Save Model
# =============================

joblib.dump(model, "best_model.pkl")
print("\nImproved model saved as best_model.pkl")

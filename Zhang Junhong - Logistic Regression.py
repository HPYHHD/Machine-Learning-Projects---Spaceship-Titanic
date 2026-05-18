# =========================================================
# SPACESHIP TITANIC - OPTIMIZED LOGISTIC REGRESSION
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Save PassengerId
test_ids = test["PassengerId"]

# =========================
# FEATURE ENGINEERING
# =========================

def feature_engineering(df):

    # ---------------------
    # Cabin Features
    # ---------------------
    df["Cabin"].fillna("Unknown/0/Unknown", inplace=True)

    cabin_split = df["Cabin"].str.split("/", expand=True)

    df["CabinDeck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2]

    # ---------------------
    # Name Features
    # ---------------------
    df["Surname"] = df["Name"].fillna("Unknown").apply(lambda x: x.split()[-1])

    # ---------------------
    # Group Features
    # ---------------------
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])

    group_counts = df["Group"].value_counts()

    df["GroupSize"] = df["Group"].map(group_counts)

    # ---------------------
    # Spending Features
    # ---------------------
    spending_cols = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck"
    ]

    for col in spending_cols:
        df[col] = df[col].fillna(0)

    df["TotalSpending"] = df[spending_cols].sum(axis=1)

    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)

    # ---------------------
    # Age Features
    # ---------------------
    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0,12,18,25,40,60,100],
        labels=[
            "Child",
            "Teen",
            "YoungAdult",
            "Adult",
            "MiddleAge",
            "Senior"
        ]
    )

    # ---------------------
    # VIP & CryoSleep
    # ---------------------
    df["VIP"] = df["VIP"].fillna(False)
    df["CryoSleep"] = df["CryoSleep"].fillna(False)

    # ---------------------
    # HomePlanet & Destination
    # ---------------------
    df["HomePlanet"] = df["HomePlanet"].fillna(
        df["HomePlanet"].mode()[0]
    )

    df["Destination"] = df["Destination"].fillna(
        df["Destination"].mode()[0]
    )

    # ---------------------
    # Outlier Handling
    # ---------------------
    for col in spending_cols + ["TotalSpending", "CabinNum"]:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)

        df[col] = df[col].clip(q1, q99)

    # ---------------------
    # Drop Unused Columns
    # ---------------------
    drop_cols = [
        "PassengerId",
        "Name",
        "Cabin",
        "Group"
    ]

    df.drop(columns=drop_cols, inplace=True)

    return df

# Apply feature engineering
train = feature_engineering(train)
test = feature_engineering(test)

# =========================
# SPLIT FEATURES / TARGET
# =========================

X = train.drop("Transported", axis=1)
y = train["Transported"].astype(int)

X_test = test.copy()

# =========================
# CATEGORICAL / NUMERICAL
# =========================

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object", "category", "bool"]
).columns.tolist()

# =========================
# PREPROCESSING
# =========================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =========================
# LOGISTIC REGRESSION
# =========================

lr = LogisticRegression(
    max_iter=5000,
    random_state=42
)

# =========================
# PIPELINE
# =========================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", lr)
    ]
)

# =========================
# HYPERPARAMETER TUNING
# =========================

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 3, 5, 10],
    "classifier__penalty": ["l2"],
    "classifier__solver": ["lbfgs", "liblinear"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# Train
grid_search.fit(X, y)

# =========================
# BEST MODEL
# =========================

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Accuracy:")
print(grid_search.best_score_)

# =========================
# CROSS VALIDATION SCORE
# =========================

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(cv_scores)

print("\nMean CV Accuracy:")
print(cv_scores.mean())

# =========================
# TRAIN FINAL MODEL
# =========================

best_model.fit(X, y)

# =========================
# PREDICT TEST
# =========================

predictions = best_model.predict(X_test)

# Convert to boolean
predictions = predictions.astype(bool)

# =========================
# CREATE SUBMISSION
# =========================

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": predictions
})

submission.to_csv("submission_LR_good.csv", index=False)

print("\nsubmission.csv generated successfully!")
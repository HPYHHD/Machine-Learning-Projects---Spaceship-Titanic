# =========================================================
# SPACESHIP TITANIC - XGBOOST + OPTUNA + 5-FOLD CV
# IMPROVED VERSION (based on data analysis)
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import optuna

# =========================================================
# LOAD DATA
# =========================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# =========================================================
# SAVE IDS
# =========================================================

test_ids = test["PassengerId"]

# =========================================================
# COMBINE TRAIN + TEST
# =========================================================

data = pd.concat([train, test], axis=0).reset_index(drop=True)

# =========================================================
# FEATURE ENGINEERING (no fitting from data distribution)
# =========================================================

# -------------------------
# CABIN FEATURES
# -------------------------

data[['CabinDeck', 'CabinNum', 'CabinSide']] = (
    data['Cabin']
    .str.split('/', expand=True)
)

data['CabinNum'] = pd.to_numeric(data['CabinNum'], errors='coerce')

# -------------------------
# GROUP FEATURES
# -------------------------

data[['GroupID', 'GroupNumber']] = (
    data['PassengerId']
    .str.split('_', expand=True)
)

group_counts = data['GroupID'].value_counts()
data['GroupSize'] = data['GroupID'].map(group_counts)
data['IsAlone'] = (data['GroupSize'] == 1).astype(int)

# -------------------------
# SPENDING FEATURES
# -------------------------

spend_cols = [
    'RoomService',
    'FoodCourt',
    'ShoppingMall',
    'Spa',
    'VRDeck'
]

# Fill spend NaN with 0 temporarily
for col in spend_cols:
    data[col] = data[col].fillna(0)

data['TotalSpend'] = data[spend_cols].sum(axis=1)

# -------------------------
# CRYOSLEEP INFERENCE (bidirectional with spending)
# -------------------------

# CryoSleep = True implies zero spending (deterministic relationship)
# If spending > 0, passenger cannot be in CryoSleep
data.loc[
    (data['CryoSleep'].isna()) & (data['TotalSpend'] > 0),
    'CryoSleep'
] = 'False'

# If spending == 0 and CryoSleep is NaN, likely in CryoSleep
data.loc[
    (data['CryoSleep'].isna()) & (data['TotalSpend'] == 0),
    'CryoSleep'
] = 'True'

# Fix inconsistencies: CryoSleep=True but spending > 0 → force spending to 0
cryo_mask = (data['CryoSleep'] == 'True') | (data['CryoSleep'] == True)
for col in spend_cols:
    data.loc[cryo_mask, col] = 0

# Recompute TotalSpend after corrections
data['TotalSpend'] = data[spend_cols].sum(axis=1)

# -------------------------
# DERIVED SPENDING FEATURES (before log transform)
# -------------------------

data['NoSpend'] = (data['TotalSpend'] == 0).astype(int)

# Number of spend categories used (0-5)
data['SpendCategoryCount'] = (data[spend_cols] > 0).sum(axis=1)

# RoomService ratio (strongest negative spend predictor)
data['RoomServiceRatio'] = np.where(
    data['TotalSpend'] > 0,
    data['RoomService'] / data['TotalSpend'],
    0
)

# Combined Spa + VRDeck (two strongest negative spend predictors)
data['SpaVRDeck'] = data['Spa'] + data['VRDeck']

# Average spend per group member
data['AvgSpendPerPerson'] = data['TotalSpend'] / data['GroupSize'].clip(lower=1)

# -------------------------
# LOG TRANSFORM (before KNN, for better distance-based imputation)
# -------------------------

for col in spend_cols + ['TotalSpend', 'SpaVRDeck', 'AvgSpendPerPerson']:
    data[col] = np.log1p(data[col])

# -------------------------
# AGE FEATURES
# -------------------------

data['AgeGroup'] = pd.cut(
    data['Age'],
    bins=[0, 12, 18, 25, 40, 60, 100],
    labels=False
)

# CryoSleep * Age interaction (CryoSleep passengers might have different age patterns)
# Will be computed after label encoding, or we can use IsCryoSleep binary flag
data['IsCryoSleep'] = ((data['CryoSleep'] == 'True') | (data['CryoSleep'] == True)).astype(int)

# =========================================================
# DROP UNUSED COLUMNS
# =========================================================

data.drop([
    'Cabin',
    'Name',
    'GroupNumber',
    'PassengerId'
], axis=1, inplace=True)

# =========================================================
# CATEGORICAL / NUMERICAL COLUMNS
# =========================================================

categorical_cols = [
    'HomePlanet',
    'CryoSleep',
    'Destination',
    'VIP',
    'CabinDeck',
    'CabinSide',
    'GroupID'
]

numerical_cols = [
    'Age',
    'RoomService',
    'FoodCourt',
    'ShoppingMall',
    'Spa',
    'VRDeck',
    'CabinNum',
    'GroupSize',
    'TotalSpend',
    'SpaVRDeck',
    'RoomServiceRatio',
    'AvgSpendPerPerson',
    'SpendCategoryCount'
]

# =========================================================
# SPLIT TRAIN / TEST
# =========================================================

train_processed = data.iloc[:len(train)].copy()
test_processed = data.iloc[len(train):].copy()

# =========================================================
# FIT ON TRAIN ONLY, TRANSFORM BOTH
# =========================================================

# --- Fill categorical NaN with train mode ---
for col in categorical_cols:
    mode_val = train_processed[col].mode()[0]
    train_processed[col] = train_processed[col].fillna(mode_val)
    test_processed[col] = test_processed[col].fillna(mode_val)

# --- Map unseen test categories to train mode ---
for col in categorical_cols:
    train_vals = set(train_processed[col].unique())
    mode_val = train_processed[col].mode()[0]
    mask = ~test_processed[col].isin(train_vals)
    test_processed.loc[mask, col] = mode_val

# --- Label encoding (fit on train) ---
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_processed[col] = le.fit_transform(train_processed[col].astype(str))
    test_processed[col] = le.transform(test_processed[col].astype(str))
    label_encoders[col] = le

# --- KNN imputer (fit on train) ---
imputer = KNNImputer(n_neighbors=5)
train_processed[numerical_cols] = imputer.fit_transform(train_processed[numerical_cols])
test_processed[numerical_cols] = imputer.transform(test_processed[numerical_cols])

# --- Fill AgeGroup NaN with train mode ---
agegroup_mode = train_processed['AgeGroup'].mode()[0]
train_processed['AgeGroup'] = train_processed['AgeGroup'].fillna(agegroup_mode)
test_processed['AgeGroup'] = test_processed['AgeGroup'].fillna(agegroup_mode)

# --- LuxuryLevel qcut (fit on train, apply bins to test) ---
train_processed['LuxuryLevel'], luxury_bins = pd.qcut(
    train_processed['TotalSpend'],
    q=4,
    labels=False,
    duplicates='drop',
    retbins=True
)
test_processed['LuxuryLevel'] = pd.cut(
    test_processed['TotalSpend'],
    bins=luxury_bins,
    labels=False,
    include_lowest=True
)
test_processed['LuxuryLevel'] = test_processed['LuxuryLevel'].fillna(
    train_processed['LuxuryLevel'].mode()[0]
)

# =========================================================
# FINAL CLEANING
# =========================================================

train_processed['LuxuryLevel'] = train_processed['LuxuryLevel'].astype(int)
test_processed['LuxuryLevel'] = test_processed['LuxuryLevel'].astype(int)
train_processed['AgeGroup'] = train_processed['AgeGroup'].astype(int)
test_processed['AgeGroup'] = test_processed['AgeGroup'].astype(int)

# =========================================================
# TARGET
# =========================================================

X = train_processed.drop('Transported', axis=1)
y = train_processed['Transported'].astype(int)

X_test = test_processed.drop('Transported', axis=1)

print(f"X shape: {X.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Features ({len(X.columns)}): {list(X.columns)}")

# =========================================================
# 5-FOLD CV
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =========================================================
# OPTUNA OBJECTIVE FUNCTION (manual CV with early stopping)
# =========================================================

def objective(trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_bin': trial.suggest_int('max_bin', 128, 512, step=64),

        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': 1,
    }

    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))

    return np.mean(scores)


# =========================================================
# RUN OPTUNA
# =========================================================

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)

study.optimize(
    objective,
    n_trials=50,
    show_progress_bar=True
)

# =========================================================
# BEST PARAMETERS
# =========================================================

print("\n==============================")
print("BEST CV SCORE:")
print(study.best_value)

print("\nBEST PARAMETERS:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# =========================================================
# TRAIN FINAL MODEL
# =========================================================

best_params = study.best_params

final_model = XGBClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    n_jobs=1
)

final_model.fit(X, y)

# =========================================================
# CHECK TRAINING SCORE (for overfitting detection)
# =========================================================

train_pred = final_model.predict(X)
train_acc = accuracy_score(y, train_pred)
print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"CV Accuracy:       {study.best_value:.4f}")
print(f"Gap (Train - CV):  {train_acc - study.best_value:.4f}")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 15 FEATURE IMPORTANCE:")
print(importance_df.head(15).to_string(index=False))

# =========================================================
# PREDICT TEST
# =========================================================

predictions = final_model.predict(X_test)

# =========================================================
# CREATE SUBMISSION
# =========================================================

submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': predictions.astype(bool)
})

submission.to_csv("submission.csv", index=False)

# =========================================================
# DONE
# =========================================================

print("\nsubmission.csv successfully created!")
print(submission.head())
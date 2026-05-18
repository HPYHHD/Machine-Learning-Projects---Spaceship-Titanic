"""
Spaceship Titanic - Pure Training Weighted Ensemble Model
Public LB reference in our experiments: about 0.80664

Model idea:
- Feature engineering on PassengerId, Cabin, spending columns, CryoSleep, route and group information
- 3-fold StratifiedKFold OOF validation
- Weighted ensemble:
    CatBoost  : 0.50
    LightGBM  : 0.30
    XGBoost   : 0.20
- Final threshold: 0.46

Input files expected in the same folder by default:
    train.csv
    test.csv
    sample_submission.csv

Output:
    submission_pure_weighted_thr46.csv

If some packages are missing, install them first, for example:
    pip install catboost lightgbm xgboost scikit-learn pandas numpy
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score

try:
    from catboost import CatBoostClassifier
except Exception as e:
    raise ImportError("CatBoost is not installed. Please run: pip install catboost") from e

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise ImportError("LightGBM is not installed. Please run: pip install lightgbm") from e

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("XGBoost is not installed. Please run: pip install xgboost") from e

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 3
THRESHOLD = 0.46

MODEL_WEIGHTS = {
    "cat": 0.50,
    "lgb": 0.30,
    "xgb": 0.20,
}

SPEND_COLS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]


def safe_mode(series: pd.Series):
    """Return mode value. If mode is empty, return np.nan."""
    mode_values = series.dropna().mode()
    if len(mode_values) == 0:
        return np.nan
    return mode_values.iloc[0]


def add_group_features(full: pd.DataFrame) -> pd.DataFrame:
    """Extract information from PassengerId."""
    passenger_parts = full["PassengerId"].astype(str).str.split("_", expand=True)
    full["GroupId"] = passenger_parts[0]
    full["GroupMemberNo"] = pd.to_numeric(passenger_parts[1], errors="coerce")

    full["GroupSize"] = full.groupby("GroupId")["PassengerId"].transform("count")
    full["IsAlone"] = (full["GroupSize"] == 1).astype(int)
    full["IsGroup"] = (full["GroupSize"] > 1).astype(int)

    return full


def add_cabin_features(full: pd.DataFrame) -> pd.DataFrame:
    """Split Cabin into Deck, Number and Side."""
    full["CabinMissing"] = full["Cabin"].isna().astype(int)

    cabin_parts = full["Cabin"].astype(str).str.split("/", expand=True)
    if cabin_parts.shape[1] < 3:
        # Safety fallback if unexpected format appears.
        for i in range(cabin_parts.shape[1], 3):
            cabin_parts[i] = np.nan

    full["CabinDeck"] = cabin_parts[0].replace("nan", np.nan)
    full["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    full["CabinSide"] = cabin_parts[2].replace("nan", np.nan)

    full["CabinNumMissing"] = full["CabinNum"].isna().astype(int)

    # Median imputation for binning only; the original missing indicator is kept.
    cabin_num_filled = full["CabinNum"].fillna(full["CabinNum"].median())
    full["CabinNumBin"] = pd.qcut(
        cabin_num_filled,
        q=6,
        labels=False,
        duplicates="drop",
    ).astype(float)

    # Coarser position feature. It is less sparse than raw CabinNum.
    full["DeckSide"] = full["CabinDeck"].astype(str) + "_" + full["CabinSide"].astype(str)

    return full


def add_name_family_features(full: pd.DataFrame) -> pd.DataFrame:
    """Use Name to build a family-size feature, but do not keep raw Surname."""
    full["NameMissing"] = full["Name"].isna().astype(int)

    # In Spaceship Titanic, names are usually formatted like 'First Last'.
    # The last token is used as a conservative surname proxy.
    full["Surname"] = full["Name"].fillna("Unknown Unknown").astype(str).str.split().str[-1]
    full.loc[full["Name"].isna(), "Surname"] = "Unknown"

    full["FamilySize"] = full.groupby("Surname")["PassengerId"].transform("count")
    full["FamilySize"] = full["FamilySize"].clip(upper=10)
    full["IsFamily"] = (full["FamilySize"] > 1).astype(int)

    return full


def add_spending_features(full: pd.DataFrame) -> pd.DataFrame:
    """Build spending-level and spending-structure features."""
    for col in SPEND_COLS:
        full[f"{col}Missing"] = full[col].isna().astype(int)

    full["SpendMissingCount"] = full[[f"{col}Missing" for col in SPEND_COLS]].sum(axis=1)

    # Spending missing values are filled with 0 because no spending is a meaningful state,
    # especially for CryoSleep passengers.
    for col in SPEND_COLS:
        full[col] = full[col].fillna(0)

    full["TotalSpend"] = full[SPEND_COLS].sum(axis=1)
    full["NoSpend"] = (full["TotalSpend"] == 0).astype(int)

    full["LuxurySpend"] = full["Spa"] + full["VRDeck"]
    full["ServiceSpend"] = full["RoomService"] + full["FoodCourt"] + full["ShoppingMall"]
    full["HasLuxurySpend"] = (full["LuxurySpend"] > 0).astype(int)
    full["HasServiceSpend"] = (full["ServiceSpend"] > 0).astype(int)

    # Log transform reduces the impact of extreme spending outliers.
    for col in SPEND_COLS + ["TotalSpend", "LuxurySpend", "ServiceSpend"]:
        full[f"Log_{col}"] = np.log1p(full[col])

    # Spending ratios describe spending structure rather than absolute amount.
    denom = full["TotalSpend"] + 1.0
    for col in SPEND_COLS:
        full[f"{col}Ratio"] = full[col] / denom

    full["LuxuryRatio"] = full["LuxurySpend"] / denom
    full["ServiceRatio"] = full["ServiceSpend"] / denom

    return full


def add_age_features(full: pd.DataFrame) -> pd.DataFrame:
    """Build age-related features and impute Age."""
    full["AgeMissing"] = full["Age"].isna().astype(int)

    # Hierarchical median imputation: HomePlanet median first, then global median.
    full["Age"] = full.groupby("HomePlanet")["Age"].transform(lambda s: s.fillna(s.median()))
    full["Age"] = full["Age"].fillna(full["Age"].median())

    full["AgeBin"] = pd.cut(
        full["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 65, 100],
        labels=False,
    ).astype(float)

    full["IsChild"] = (full["Age"] < 13).astype(int)
    full["IsTeen"] = ((full["Age"] >= 13) & (full["Age"] < 18)).astype(int)
    full["IsAdult"] = ((full["Age"] >= 18) & (full["Age"] < 60)).astype(int)
    full["IsSenior"] = (full["Age"] >= 60).astype(int)

    return full


def add_interaction_features(full: pd.DataFrame) -> pd.DataFrame:
    """Build stable categorical interaction features."""
    # Convert to string so missing values become explicit categories in interactions.
    hp = full["HomePlanet"].astype(str)
    dest = full["Destination"].astype(str)
    deck = full["CabinDeck"].astype(str)
    side = full["CabinSide"].astype(str)
    cryo = full["CryoSleep"].astype(str)
    nospend = full["NoSpend"].astype(str)

    full["HomeDest"] = hp + "_" + dest
    full["HomeDeck"] = hp + "_" + deck
    full["DeckDest"] = deck + "_" + dest
    full["DeckSideDest"] = deck + "_" + side + "_" + dest
    full["CryoNoSpend"] = cryo + "_" + nospend
    full["CryoHome"] = cryo + "_" + hp
    full["CryoDeck"] = cryo + "_" + deck

    return full


def impute_groupwise_categories(full: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative categorical imputation.
    For some variables, passengers in the same group often share the same value.
    Use group mode first, then global mode.
    """
    for col in ["HomePlanet", "Destination", "CryoSleep", "VIP"]:
        if col not in full.columns:
            continue

        # Fill using GroupId mode when possible.
        group_modes = full.groupby("GroupId")[col].transform(lambda s: safe_mode(s))
        full[col] = full[col].fillna(group_modes)

        # Fill remaining missing values using global mode.
        global_mode = safe_mode(full[col])
        full[col] = full[col].fillna(global_mode)

    return full


def engineer_features(train: pd.DataFrame, test: pd.DataFrame):
    """Full feature engineering pipeline."""
    train_no_target = train.drop(columns=["Transported"]).copy()
    test_copy = test.copy()

    full = pd.concat([train_no_target, test_copy], axis=0, ignore_index=True)

    full = add_group_features(full)
    full = impute_groupwise_categories(full)
    full = add_cabin_features(full)
    full = add_name_family_features(full)
    full = add_spending_features(full)
    full = add_age_features(full)
    full = add_interaction_features(full)

    # Boolean columns can contain True/False/NaN; convert them to strings for stable encoding.
    for col in ["CryoSleep", "VIP"]:
        if col in full.columns:
            full[col] = full[col].astype(str)

    # Drop raw identifiers and very high-cardinality raw fields.
    # GroupId and Surname are used only to create GroupSize / FamilySize.
    drop_cols = [
        "PassengerId",
        "Cabin",
        "Name",
        "GroupId",
        "Surname",
    ]
    full = full.drop(columns=[col for col in drop_cols if col in full.columns])

    # Encode categorical columns consistently over train + test.
    cat_cols = full.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        full[col] = full[col].fillna("Unknown").astype(str)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    if len(cat_cols) > 0:
        full[cat_cols] = encoder.fit_transform(full[cat_cols])

    # Fill any remaining numeric missing values with median.
    for col in full.columns:
        if full[col].isna().any():
            full[col] = full[col].fillna(full[col].median())

    # Ensure all columns are numeric.
    full = full.astype(float)

    X = full.iloc[: len(train)].copy()
    X_test = full.iloc[len(train) :].copy()

    return X, X_test, cat_cols


def build_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=550,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=5.0,
        random_strength=0.8,
        loss_function="Logloss",
        eval_metric="Accuracy",
        bootstrap_type="Bernoulli",
        subsample=0.85,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def build_lightgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=650,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=6,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgboost(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=520,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.03,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def train_weighted_ensemble(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
    """Train CatBoost, LightGBM, XGBoost using StratifiedKFold and blend them."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = {
        "cat": np.zeros(len(X)),
        "lgb": np.zeros(len(X)),
        "xgb": np.zeros(len(X)),
    }
    test_pred = {
        "cat": np.zeros(len(X_test)),
        "lgb": np.zeros(len(X_test)),
        "xgb": np.zeros(len(X_test)),
    }

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_seed = SEED + fold * 100

        cat_model = build_catboost(fold_seed)
        lgb_model = build_lightgbm(fold_seed)
        xgb_model = build_xgboost(fold_seed)

        # CatBoost
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=False)
        oof["cat"][val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_pred["cat"] += cat_model.predict_proba(X_test)[:, 1] / N_SPLITS
        print("CatBoost fold acc:", accuracy_score(y_val, oof["cat"][val_idx] >= 0.5))

        # LightGBM
        lgb_model.fit(X_tr, y_tr)
        oof["lgb"][val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        test_pred["lgb"] += lgb_model.predict_proba(X_test)[:, 1] / N_SPLITS
        print("LightGBM fold acc:", accuracy_score(y_val, oof["lgb"][val_idx] >= 0.5))

        # XGBoost
        xgb_model.fit(X_tr, y_tr)
        oof["xgb"][val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_pred["xgb"] += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS
        print("XGBoost fold acc:", accuracy_score(y_val, oof["xgb"][val_idx] >= 0.5))

    # Model-level OOF accuracy at threshold 0.50.
    print("\n========== Single Model OOF Accuracy @ 0.50 ==========")
    for name in ["cat", "lgb", "xgb"]:
        acc = accuracy_score(y, oof[name] >= 0.5)
        print(f"{name}: {acc:.5f}")

    weighted_oof = sum(MODEL_WEIGHTS[name] * oof[name] for name in MODEL_WEIGHTS)
    weighted_test = sum(MODEL_WEIGHTS[name] * test_pred[name] for name in MODEL_WEIGHTS)

    print("\n========== Weighted Blend OOF Accuracy ==========")
    for thr in [0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]:
        acc = accuracy_score(y, weighted_oof >= thr)
        true_count = int((weighted_test >= thr).sum())
        print(f"threshold={thr:.2f} | OOF acc={acc:.5f} | test True count={true_count}")

    final_acc = accuracy_score(y, weighted_oof >= THRESHOLD)
    print(f"\nFinal selected threshold: {THRESHOLD}")
    print(f"Weighted OOF accuracy @ {THRESHOLD}: {final_acc:.5f}")

    return weighted_oof, weighted_test, oof, test_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test.csv")
    parser.add_argument("--sample", type=str, default="sample_submission.csv", help="Path to sample_submission.csv")
    parser.add_argument("--output", type=str, default="submission_pure_weighted_thr46.csv", help="Output submission path")
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    sample_path = Path(args.sample)

    print("Reading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)

    y = train["Transported"].astype(int)

    print("Engineering features...")
    X, X_test, cat_cols = engineer_features(train, test)
    print(f"Train shape: {X.shape}")
    print(f"Test shape : {X_test.shape}")
    print(f"Categorical columns encoded: {len(cat_cols)}")

    print("Training weighted ensemble...")
    weighted_oof, weighted_test, oof, test_pred = train_weighted_ensemble(X, y, X_test)

    final_pred = weighted_test >= THRESHOLD

    submission = sample_submission.copy()
    submission["Transported"] = final_pred.astype(bool)
    submission.to_csv(args.output, index=False)

    print("\n========== Submission Saved ==========")
    print(f"Output file: {args.output}")
    print(f"Rows: {len(submission)}")
    print(f"True : {int(submission['Transported'].sum())}")
    print(f"False: {int((~submission['Transported']).sum())}")
    print(submission.head())


if __name__ == "__main__":
    main()

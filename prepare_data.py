"""
data/prepare_data.py
---------------------
PURPOSE:
  Loads the dataset from HuggingFace, cleans it, and handles class imbalance
  using SMOTE before saving a train-ready CSV.

WHY SMOTE (Synthetic Minority Oversampling TEchnique):
  Our primary_complication column is heavily skewed:
    none                 57%  ← dominates
    gestational_diabetes 14%
    preeclampsia         12%
    severe_anemia        10%
    hemorrhage            6%
    eclampsia             2%  ← model will almost never learn this

  SMOTE fixes this by GENERATING synthetic new rows for minority classes
  using interpolation between real existing examples.
  
  Result after SMOTE: all 6 classes have equal representation.
  
  WHY NOT just duplicate rows (oversampling)?
    Duplicating creates exact copies → model memorizes them → overfits.
    SMOTE creates NEW plausible examples → model generalizes better.

  WHY NOT undersample the majority?
    We'd throw away 80% of our "none" rows → waste of 30,000 real data points.

IMPORTANT: SMOTE is applied ONLY to the training set, never the test set.
  Applying it to test data would be data leakage — you'd be testing on
  synthetic data you helped create, which gives falsely optimistic metrics.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

# Output paths — relative to this file
import os as _os
OUT_DIR       = _os.path.dirname(_os.path.abspath(__file__))
TRAIN_PATH    = _os.path.join(OUT_DIR, "train.csv")
TEST_PATH     = _os.path.join(OUT_DIR, "test.csv")
ENCODERS_PATH = _os.path.join(OUT_DIR, "encoders.pkl")


# ─── FEATURE CONFIGURATION ────────────────────────────────────────────────────
# These are the columns we feed into the model.
# We exclude: id, delivery_mode, pregnancy_outcome (not available at prediction time)
# We include: anemia_status as an encoded feature (it's clinically informative)

NUMERIC_FEATURES = [
    "age_years",
    "gravidity",
    "parity",
    "gestational_age_weeks",
    "bmi_pre_pregnancy",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "hemoglobin_gdl",
    "fasting_glucose_mgdl",
    "proteinuria",          # 0/1 binary
    "hiv_status",           # 0/1 binary
    "anc_visits",
]

CATEGORICAL_FEATURES = [
    "anemia_status",        # none / mild / moderate / severe
]

# What we're predicting
TARGET_COMPLICATION = "primary_complication"
TARGET_RISK         = "risk_level"


def load_raw_data() -> pd.DataFrame:
    """
    Loads the electricsheep dataset from HuggingFace.
    The 'high_burden' config is the main clinical dataset.
    """
    print("📥 Loading dataset from HuggingFace...")
    ds = load_dataset("electricsheepafrica/maternal-health-pregnancy", "high_burden")
    df = ds["train"].to_pandas()
    print(f"   Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe:
    1. Standardizes string columns to lowercase + strip whitespace
    2. Handles missing values safely
    3. Removes clearly impossible values (e.g. BP of 0)
    """
    print("\n🧹 Cleaning data...")

    # Standardize all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.lower().str.strip()

    # ── Handle missing values ──────────────────────────────────────
    # Numeric: fill with column median (robust to outliers)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"   Filled {missing} missing values in '{col}' with median {median_val:.1f}")

    # Categorical: fill with "none" (most common in anemia_status)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("none")

    # ── Remove impossible values ───────────────────────────────────
    # A systolic BP of 0 or BMI of 0 is a data entry error, not a real patient
    df = df[df["systolic_bp_mmhg"] > 0]
    df = df[df["diastolic_bp_mmhg"] > 0]
    df = df[df["bmi_pre_pregnancy"] > 0]

    print(f"   Rows after cleaning: {len(df):,}")
    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encodes categorical features and target columns into numbers.
    
    Returns:
        df:       DataFrame with encoded columns
        encoders: dict of fitted LabelEncoders (saved for inference time)
    
    WHY SAVE ENCODERS:
        When a nurse enters vitals in the app, we need to transform
        "moderate" anemia_status into the same number the model trained on.
        Without saving the encoder, we'd have to guess — and get it wrong.
    """
    encoders = {}

    # ── Encode anemia_status (feature) ────────────────────────────
    le_anemia = LabelEncoder()
    df["anemia_status_enc"] = le_anemia.fit_transform(df["anemia_status"])
    encoders["anemia_status"] = le_anemia
    print(f"\n   anemia_status classes: {list(le_anemia.classes_)}")

    # ── Encode primary_complication (target 1) ────────────────────
    le_comp = LabelEncoder()
    df["complication_enc"] = le_comp.fit_transform(df[TARGET_COMPLICATION])
    encoders["primary_complication"] = le_comp
    print(f"   primary_complication classes: {list(le_comp.classes_)}")

    # ── Encode risk_level (target 2) ─────────────────────────────
    # Map manually to preserve clinical ordering: low=0, moderate=1, high=2
    risk_map = {"low": 0, "moderate": 1, "high": 2}
    df["risk_enc"] = df[TARGET_RISK].map(risk_map)
    encoders["risk_level"] = risk_map
    print(f"   risk_level mapping: {risk_map}")

    return df, encoders


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, label: str) -> tuple:
    """
    Applies SMOTE to balance the training set.
    
    SMOTE works by:
      1. For each minority class sample, find its k nearest neighbors
      2. Randomly interpolate between the sample and a neighbor
      3. This creates a new synthetic sample that's "between" real ones
    
    k_neighbors=5 is the standard default.
    We use k_neighbors=3 for eclampsia (2% = ~600 rows) because with very
    few samples, 5 neighbors might not all exist.
    
    Args:
        X_train: feature matrix (training only)
        y_train: target labels (training only)
        label:   string name for logging
    
    Returns:
        X_resampled, y_resampled (balanced)
    """
    print(f"\n⚖️  Applying SMOTE to balance '{label}' classes...")
    print(f"   Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Use k_neighbors=3 to be safe with very small minority classes
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"   After SMOTE:  {dict(zip(*np.unique(y_res, return_counts=True)))}")
    print(f"   New training size: {len(X_res):,} rows")
    return X_res, y_res


def main():
    # ── Step 1: Load ──────────────────────────────────────────────
    df = load_raw_data()

    # ── Step 2: Clean ─────────────────────────────────────────────
    df = clean_data(df)

    # ── Step 3: Encode ────────────────────────────────────────────
    df, encoders = encode_features(df)

    # ── Step 4: Build feature matrix ──────────────────────────────
    # Final feature list: all numeric + the encoded anemia_status
    all_features = NUMERIC_FEATURES + ["anemia_status_enc"]

    X = df[all_features].copy()
    y_comp = df["complication_enc"]   # Target 1: complication
    y_risk = df["risk_enc"]           # Target 2: risk level

    # ── Step 5: Train/Test split BEFORE SMOTE ─────────────────────
    # Critical: SMOTE only on train set. Test set stays real/unmodified.
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_comp, y_risk,
        test_size=0.2,
        random_state=42,
        stratify=y_comp    # Preserve class proportions in both splits
    )

    print(f"\n✂️  Split: {len(X_train):,} train / {len(X_test):,} test")

    # ── Step 6: Apply SMOTE to training set only ──────────────────
    X_train_comp, yc_train_bal = apply_smote(X_train, yc_train, "primary_complication")

    # Risk level is already balanced (38/35/26) — SMOTE not needed
    print("\n⚖️  Risk level already balanced — skipping SMOTE")

    # ── Step 7: Save processed data ───────────────────────────────
    # Reassemble train DataFrames
    train_comp_df = pd.DataFrame(X_train_comp, columns=all_features)
    train_comp_df["complication_enc"] = yc_train_bal.values

    # For risk model: use original (unSMOTEd) train split
    train_risk_df = pd.DataFrame(X_train.values, columns=all_features)
    train_risk_df["risk_enc"] = yr_train.values

    test_df = pd.DataFrame(X_test.values, columns=all_features)
    test_df["complication_enc"] = yc_test.values
    test_df["risk_enc"] = yr_test.values

    train_comp_df.to_csv(os.path.join(OUT_DIR, "train_complication.csv"), index=False)
    train_risk_df.to_csv(os.path.join(OUT_DIR, "train_risk.csv"), index=False)
    test_df.to_csv(TEST_PATH, index=False)

    # Save encoders (needed by the app at inference time)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(all_features, os.path.join(OUT_DIR, "feature_names.pkl"))

    print(f"\n✅ Saved training data and encoders to {OUT_DIR}/")
    print("   Next step: python models/train_models.py")


if __name__ == "__main__":
    main()

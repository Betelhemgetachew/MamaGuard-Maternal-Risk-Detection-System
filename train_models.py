print("RUNNING FROM:", __file__)
"""
models/train_models.py
-----------------------
PURPOSE:
  Trains TWO XGBoost models and saves them:
    1. complication_model.pkl  — predicts primary_complication (6 classes)
    2. risk_model.pkl          — predicts risk_level (3 classes)

WHY TWO SEPARATE MODELS:
  They answer different clinical questions and have different class structures.
  Training them separately means:
    - Each model is optimized for its own task
    - If one needs retraining (e.g. new complication type added), 
      the other is unaffected
    - Easier to evaluate and debug independently

MODEL EVALUATION — WHAT METRICS WE USE:
  We do NOT just use accuracy. Here's why:

  Accuracy is misleading on clinical data:
    If model always predicts "none", it gets 57% accuracy.
    That's useless — it never catches eclampsia.

  Instead we use:
    - Recall (Sensitivity): Of all real eclampsia cases, how many did we catch?
      This is the most important metric for dangerous conditions.
      Missing eclampsia = patient dies. False alarm = extra test. Recall wins.
    
    - Precision: Of all the times we flagged eclampsia, how often were we right?
      Important for nurse trust — too many false alarms = nurses ignore alerts.
    
    - F1 Score: Harmonic mean of precision and recall. Our primary metric.
    
    - Macro F1: Average F1 across ALL classes (including rare ones like eclampsia).
      This penalizes the model if it ignores minority classes.

RUN THIS AFTER: python data/prepare_data.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
import json

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = BASE_DIR
MODEL_DIR = BASE_DIR

TRAIN_COMP_PATH = os.path.join(BASE_DIR, "train_complication.csv")
TRAIN_RISK_PATH = os.path.join(BASE_DIR, "train_risk.csv")
TEST_PATH       = os.path.join(BASE_DIR, "test.csv")

ENCODERS_PATH   = os.path.join(BASE_DIR, "encoders.pkl")
FEATURES_PATH   = os.path.join(BASE_DIR, "feature_names.pkl")

COMP_MODEL_PATH = os.path.join(BASE_DIR, "complication_model.pkl")
RISK_MODEL_PATH = os.path.join(BASE_DIR, "risk_model.pkl")

METRICS_PATH    = os.path.join(BASE_DIR, "model_metrics.json")


def build_xgboost(n_classes: int, scale_weights: dict = None) -> xgb.XGBClassifier:
    """
    Builds an XGBoost classifier with settings tuned for clinical tabular data.
    
    PARAMETER EXPLANATIONS:
    
    n_estimators=300:
      Number of trees. More trees = better accuracy up to a point.
      300 is good for 30k rows. Beyond ~500, returns diminish.
    
    max_depth=5:
      Max depth of each tree. Deeper = more complex patterns learned.
      5 is conservative to avoid overfitting on clinical data.
      (Medical patterns are rarely super deep — "high BP + high glucose = GDM" 
       is only depth 2)
    
    learning_rate=0.05:
      How much each new tree corrects the previous error.
      Lower = slower but more accurate. 0.05 pairs well with 300 trees.
    
    subsample=0.8:
      Each tree only sees 80% of training rows (random sample).
      This prevents overfitting by adding randomness.
    
    colsample_bytree=0.8:
      Each tree only sees 80% of features.
      Forces model to learn from different feature combinations → more robust.
    
    min_child_weight=3:
      A leaf node must have at least 3 samples to be created.
      Prevents the model from creating rules for just 1-2 patients → less overfit.
    
    scale_pos_weight (via sample_weight at fit time):
      We handle class weights manually at fit() time instead of here,
      because XGBoost's built-in only supports binary. For multi-class
      we pass sample_weight array to fit().
    """
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softprob",   # Outputs probabilities for ALL classes
        num_class=n_classes,
        random_state=42,
        verbosity=0,
        tree_method="hist",           # Faster training algorithm
    )
    return model


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Computes per-sample weights to handle class imbalance.
    
    Even after SMOTE, we add sample weights as a second layer of protection.
    This is especially important for eclampsia (most dangerous, fewest cases).
    
    Formula: weight for class c = total_samples / (n_classes * count_of_c)
    
    Example:
      1000 samples, 4 classes, 500 "none", 50 "eclampsia"
      weight for "none"     = 1000 / (4 * 500) = 0.5   (downweighted)
      weight for "eclampsia"= 1000 / (4 * 50)  = 5.0   (upweighted 10x)
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)

    class_weight = {c: total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([class_weight[label] for label in y])

    return sample_weights


def train_complication_model(feature_names: list, encoders: dict) -> dict:
    """
    Trains Model 1: primary_complication predictor.
    
    6 output classes:
      0: eclampsia
      1: gestational_diabetes
      2: hemorrhage
      3: none
      4: preeclampsia
      5: severe_anemia
    (alphabetical order from LabelEncoder)
    
    Returns dict of evaluation metrics.
    """
    print("\n" + "="*60)
    print("🧠 TRAINING MODEL 1: Primary Complication")
    print("="*60)

    # Load data
    train_df = pd.read_csv(TRAIN_COMP_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df[feature_names].values
    y_train = train_df["complication_enc"].values.astype(int)

    X_test  = test_df[feature_names].values
    y_test  = test_df["complication_enc"].values.astype(int)

    n_classes = len(np.unique(y_train))
    print(f"   Classes: {n_classes} | Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Compute sample weights (extra protection beyond SMOTE)
    sample_weights = compute_sample_weights(y_train)

    # Build and train
    model = build_xgboost(n_classes)

    print("   Training... (this takes ~30-60 seconds)")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── Evaluate ──────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    le = encoders["primary_complication"]

    print("\n📊 COMPLICATION MODEL — Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    joblib.dump(model, COMP_MODEL_PATH)
    print(f"✅ Saved to {COMP_MODEL_PATH}")

    # Return metrics for JSON log
    from sklearn.metrics import f1_score, accuracy_score
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "macro_f1":  round(f1_score(y_test, y_pred, average="macro"), 4),
        "classes":   list(le.classes_)
    }


def train_risk_model(feature_names: list, encoders: dict) -> dict:
    """
    Trains Model 2: risk_level predictor.
    
    3 output classes: low=0, moderate=1, high=2
    
    Dataset is already balanced (38/35/26) so no SMOTE needed.
    We still use sample weights as a light correction.
    """
    print("\n" + "="*60)
    print("🧠 TRAINING MODEL 2: Risk Level")
    print("="*60)

    train_df = pd.read_csv(TRAIN_RISK_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df[feature_names].values
    y_train = train_df["risk_enc"].values.astype(int)

    X_test  = test_df[feature_names].values
    y_test  = test_df["risk_enc"].values.astype(int)

    print(f"   Classes: 3 | Train: {len(X_train):,} | Test: {len(X_test):,}")

    sample_weights = compute_sample_weights(y_train)

    model = build_xgboost(n_classes=3)

    print("   Training...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── Evaluate ──────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    risk_labels = ["low", "moderate", "high"]

    print("\n📊 RISK MODEL — Classification Report:")
    print(classification_report(y_test, y_pred, target_names=risk_labels))

    joblib.dump(model, RISK_MODEL_PATH)
    print(f"✅ Saved to {RISK_MODEL_PATH}")

    from sklearn.metrics import f1_score, accuracy_score
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "classes":  risk_labels
    }


def main():
    # Verify data files exist
    for path in [TRAIN_COMP_PATH, TRAIN_RISK_PATH, TEST_PATH, ENCODERS_PATH]:
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            print("   Run: python data/prepare_data.py first")
            return

    # Load encoders and feature names
    encoders      = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    print(f"📋 Features ({len(feature_names)}): {feature_names}")

    # Train both models
    comp_metrics = train_complication_model(feature_names, encoders)
    risk_metrics = train_risk_model(feature_names, encoders)

    # Save metrics to JSON (useful for dashboard "Model Info" tab)
    metrics = {
        "complication_model": comp_metrics,
        "risk_model":         risk_metrics
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*60)
    print("✅ ALL DONE")
    print(f"   Complication model accuracy: {comp_metrics['accuracy']:.1%}")
    print(f"   Complication model macro F1: {comp_metrics['macro_f1']:.1%}")
    print(f"   Risk model accuracy:         {risk_metrics['accuracy']:.1%}")
    print(f"   Risk model macro F1:         {risk_metrics['macro_f1']:.1%}")
    print("\n🚀 Launch app: streamlit run app.py")


if __name__ == "__main__":
    main()

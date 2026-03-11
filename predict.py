"""
models/predict.py
------------------
PURPOSE:
  The inference engine — takes a nurse's vitals input and runs both models.
  This is what gets called every time the "Run AI Assessment" button is clicked.

  Returns:
    - Top-3 complication predictions with probabilities
    - Risk level with probability
    - SHAP explanations (human-readable)
    - Condition-specific suggested screenings

FLOW:
  nurse inputs vitals
        ↓
  impute_vitals()           ← fill any missing fields safely
        ↓
  build_feature_vector()    ← convert to numpy array in correct order
        ↓
  complication_model.predict_proba()  ← 6 class probabilities
  risk_model.predict_proba()          ← 3 class probabilities
        ↓
  shap_explainer()          ← which features drove the top prediction?
        ↓
  condition_screenings()    ← what should the nurse do?
        ↓
  return PredictionResult
"""

import numpy as np
import pandas as pd
import joblib
import shap
import os
from dataclasses import dataclass, field

# Paths
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMP_MODEL_PATH = os.path.join(BASE_DIR, "complication_model.pkl")
RISK_MODEL_PATH = os.path.join(BASE_DIR, "risk_model.pkl")
ENCODERS_PATH   = os.path.join(BASE_DIR, "encoders.pkl")
FEATURES_PATH   = os.path.join(BASE_DIR, "feature_names.pkl")
# ─── RESULT DATA STRUCTURE ────────────────────────────────────────────────────

@dataclass
class ComplicationPrediction:
    condition: str        # e.g. "gestational_diabetes"
    probability: float    # 0.0 to 1.0
    display_name: str     # e.g. "Gestational Diabetes"
    rank: int             # 1 = most likely


@dataclass
class PredictionResult:
    # Complication predictions (top 3)
    top_conditions: list[ComplicationPrediction] = field(default_factory=list)

    # Risk level
    risk_level: str   = ""   # "low", "moderate", "high"
    risk_prob: float  = 0.0

    # SHAP explanations
    shap_factors: list[dict] = field(default_factory=list)   # [{"feature": ..., "contribution": ...}]
    explanation_text: list[str] = field(default_factory=list) # ["Elevated BP (primary driver)", ...]

    # Suggested actions
    screenings: list[dict] = field(default_factory=list)

    # Warnings (imputed fields, missing hemoglobin, etc.)
    warnings: list[str] = field(default_factory=list)


# ─── DISPLAY NAME MAPPING ─────────────────────────────────────────────────────

CONDITION_DISPLAY = {
    "none":                 "No Complication",
    "gestational_diabetes": "Gestational Diabetes",
    "preeclampsia":         "Preeclampsia",
    "severe_anemia":        "Severe Anemia",
    "hemorrhage":           "Hemorrhage",
    "eclampsia":            "Eclampsia",
}

FEATURE_DISPLAY = {
    "age_years":              "Patient age",
    "gravidity":              "Number of pregnancies",
    "parity":                 "Number of births",
    "gestational_age_weeks":  "Gestational age",
    "bmi_pre_pregnancy":      "Pre-pregnancy BMI",
    "systolic_bp_mmhg":       "Systolic blood pressure",
    "diastolic_bp_mmhg":      "Diastolic blood pressure",
    "hemoglobin_gdl":         "Hemoglobin level",
    "fasting_glucose_mgdl":   "Fasting blood glucose",
    "proteinuria":            "Proteinuria (protein in urine)",
    "hiv_status":             "HIV status",
    "anc_visits":             "Number of ANC visits",
    "anemia_status_enc":      "Anemia severity",
}

# ─── SCREENING SUGGESTIONS ────────────────────────────────────────────────────
# Maps condition → what the nurse should do
# "Consider" and "Screen for" language — NOT "prescribe" or "diagnose"

CONDITION_SCREENINGS = {
    "gestational_diabetes": {
        "action":    "Order Oral Glucose Tolerance Test (OGTT)",
        "followup":  "Dietary counselling. Return visit in 1 week.",
        "guideline": "WHO Diagnostic Criteria for GDM (2013)",
        "urgency":   "routine"
    },
    "preeclampsia": {
        "action":    "Check urine protein (dipstick). Repeat BP in 4 hours.",
        "followup":  "If BP ≥ 160/110, refer to physician immediately.",
        "guideline": "WHO ANC Recommendation: Preeclampsia screening",
        "urgency":   "urgent"
    },
    "severe_anemia": {
        "action":    "Order full blood count (FBC). Check hemoglobin.",
        "followup":  "Prescribe oral iron (60mg/day). If Hb < 7, consider referral.",
        "guideline": "WHO: Hb < 7 g/dL = severe anemia in pregnancy",
        "urgency":   "urgent"
    },
    "hemorrhage": {
        "action":    "Assess bleeding source. Monitor vital signs closely.",
        "followup":  "Immediate physician referral if active bleeding.",
        "guideline": "WHO Recommendations for Prevention of PPH",
        "urgency":   "urgent"
    },
    "eclampsia": {
        "action":    "URGENT: Notify physician immediately.",
        "followup":  "Do not leave patient alone. Prepare for possible seizure management.",
        "guideline": "WHO: Eclampsia is a medical emergency",
        "urgency":   "critical"
    },
    "none": {
        "action":    "Continue routine antenatal care.",
        "followup":  "Schedule next visit as per ANC calendar.",
        "guideline": "WHO ANC Model: minimum 8 contacts",
        "urgency":   "routine"
    }
}


# ─── MODEL LOADER (cached) ────────────────────────────────────────────────────

_models_cache = {}

def load_models() -> tuple:
    """
    Loads both models + encoders into memory.
    Uses a module-level cache so they're only loaded once per session.
    """
    global _models_cache

    if _models_cache:
        return (
            _models_cache["comp_model"],
            _models_cache["risk_model"],
            _models_cache["encoders"],
            _models_cache["feature_names"]
        )

    if not os.path.exists(COMP_MODEL_PATH):
        raise FileNotFoundError(
            "Models not found. Run:\n"
            "  python data/prepare_data.py\n"
            "  python models/train_models.py"
        )

    comp_model    = joblib.load(COMP_MODEL_PATH)
    risk_model    = joblib.load(RISK_MODEL_PATH)
    encoders      = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    _models_cache = {
        "comp_model":    comp_model,
        "risk_model":    risk_model,
        "encoders":      encoders,
        "feature_names": feature_names
    }

    return comp_model, risk_model, encoders, feature_names


# ─── MAIN PREDICT FUNCTION ────────────────────────────────────────────────────

def predict(vitals: dict) -> PredictionResult:
    """
    Main prediction function. Takes a dict of vitals and returns a PredictionResult.
    
    Args:
        vitals: dict with keys matching feature names, e.g.:
            {
                "age_years": 28,
                "systolic_bp_mmhg": 145,
                "diastolic_bp_mmhg": 95,
                "hemoglobin_gdl": 9.5,
                "fasting_glucose_mgdl": 105,
                "proteinuria": 1,
                "anemia_status": "mild",
                ...
            }
    
    Returns:
        PredictionResult with all predictions + explanations
    """
    result = PredictionResult()

    comp_model, risk_model, encoders, feature_names = load_models()

    # ── Step 1: Encode categorical features ───────────────────────
    vitals_encoded = vitals.copy()

    # Encode anemia_status: "mild" → integer
    anemia_enc = encoders["anemia_status"]
    anemia_val = str(vitals.get("anemia_status", "none")).lower().strip()

    # Handle unseen labels gracefully
    if anemia_val not in anemia_enc.classes_:
        anemia_val = "none"
        result.warnings.append("Unrecognized anemia_status value — defaulted to 'none'")

    vitals_encoded["anemia_status_enc"] = int(
        anemia_enc.transform([anemia_val])[0]
    )

    # ── Step 2: Build feature vector in correct order ─────────────
    # Order MUST match what the model was trained on
    feature_vector = []
    imputed = []

    # Population medians as last-resort fallback
    MEDIANS = {
        "age_years": 27, "gravidity": 2, "parity": 1,
        "gestational_age_weeks": 28, "bmi_pre_pregnancy": 23,
        "systolic_bp_mmhg": 115, "diastolic_bp_mmhg": 75,
        "hemoglobin_gdl": 11.5, "fasting_glucose_mgdl": 85,
        "proteinuria": 0, "hiv_status": 0, "anc_visits": 4,
        "anemia_status_enc": 0
    }

    for feat in feature_names:
        val = vitals_encoded.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = MEDIANS.get(feat, 0)
            imputed.append(feat)
        feature_vector.append(float(val))

    if imputed:
        result.warnings.append(
            f"Missing fields filled with population median: {', '.join(imputed)}"
        )

    X = np.array(feature_vector).reshape(1, -1)
    X_df = pd.DataFrame(X, columns=feature_names)

    # ── Step 3: Predict complications (top 3) ─────────────────────
    comp_proba = comp_model.predict_proba(X)[0]  # Array of 6 probabilities
    le_comp    = encoders["primary_complication"]
    class_names = le_comp.classes_               # e.g. ["eclampsia", "gestational_diabetes", ...]

    # Sort by probability descending
    sorted_indices = np.argsort(comp_proba)[::-1]

    for rank, idx in enumerate(sorted_indices[:3], start=1):
        condition_key = class_names[idx]
        result.top_conditions.append(ComplicationPrediction(
            condition=condition_key,
            probability=round(float(comp_proba[idx]), 4),
            display_name=CONDITION_DISPLAY.get(condition_key, condition_key.replace("_", " ").title()),
            rank=rank
        ))

    # ── Step 4: Predict risk level ────────────────────────────────
    risk_proba  = risk_model.predict_proba(X)[0]  # [low_prob, mod_prob, high_prob]
    risk_labels = ["low", "moderate", "high"]
    risk_idx    = int(np.argmax(risk_proba))

    result.risk_level = risk_labels[risk_idx]
    result.risk_prob  = round(float(risk_proba[risk_idx]), 4)

    # ── Step 5: SHAP explanations for top complication ────────────
    top_condition_idx = sorted_indices[0]  # The most likely condition

    try:
        explainer  = shap.TreeExplainer(comp_model)
        shap_vals  = explainer.shap_values(X_df)

        # For multi-class: shap_vals is shape (n_classes, n_samples, n_features)
        # Get SHAP values for the top predicted class
        if isinstance(shap_vals, list):
            vals_for_class = shap_vals[top_condition_idx][0]
        else:
            vals_for_class = shap_vals[top_condition_idx]

        # Build sorted factor list
        factors = []
        for i, feat in enumerate(feature_names):
            factors.append({
                "feature":      feat,
                "display_name": FEATURE_DISPLAY.get(feat, feat),
                "shap_value":   float(vals_for_class[i]),
                "direction":    "increases" if vals_for_class[i] > 0 else "decreases",
                "abs_value":    abs(float(vals_for_class[i]))
            })

        factors.sort(key=lambda x: x["abs_value"], reverse=True)

        # Top 4 risk-increasing factors
        top_factors = [f for f in factors if f["direction"] == "increases"][:4]
        result.shap_factors = top_factors

        # Convert to text
        templates = {
            "systolic_bp_mmhg":      "Elevated systolic blood pressure",
            "diastolic_bp_mmhg":     "Elevated diastolic blood pressure",
            "fasting_glucose_mgdl":  "High fasting blood glucose",
            "hemoglobin_gdl":        "Low hemoglobin level",
            "proteinuria":           "Protein detected in urine",
            "bmi_pre_pregnancy":     "High pre-pregnancy BMI",
            "age_years":             "Patient age is a contributing factor",
            "gestational_age_weeks": "Gestational age at assessment",
            "gravidity":             "High number of previous pregnancies",
            "anemia_status_enc":     "Anemia severity",
            "anc_visits":            "Low antenatal care attendance",
            "hiv_status":            "HIV status",
        }

        texts = []
        for i, f in enumerate(top_factors):
            label = templates.get(f["feature"], f"Elevated {f['display_name']}")
            if i == 0:
                label += " (primary driver)"
            texts.append(label)

        result.explanation_text = texts

    except Exception as e:
        result.warnings.append(f"SHAP explanation unavailable: {str(e)}")
        result.explanation_text = ["Explanation unavailable — see raw prediction above"]

    # ── Step 6: Screening suggestions ─────────────────────────────
    # Add screening for top predicted condition
    top_key = result.top_conditions[0].condition if result.top_conditions else "none"
    screening = CONDITION_SCREENINGS.get(top_key, CONDITION_SCREENINGS["none"])
    result.screenings = [{"condition": top_key, **screening}]

    # Also add screening for #2 if probability > 15% (clinically significant)
    if len(result.top_conditions) > 1 and result.top_conditions[1].probability >= 0.15:
        second_key = result.top_conditions[1].condition
        if second_key != "none":
            second_screening = CONDITION_SCREENINGS.get(second_key)
            if second_screening:
                result.screenings.append({"condition": second_key, **second_screening})

    return result

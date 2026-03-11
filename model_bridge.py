"""
model_bridge.py — all predictions from Project 1's XGBoost models only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict as _p1_predict, load_models

_RISK_TO_UI = {"high": "high risk", "moderate": "mid risk", "low": "low risk"}

def load_model():
    comp_model, risk_model, encoders, feature_names = load_models()
    return comp_model, encoders.get("primary_complication")

def train_model(df=None):
    raise RuntimeError("Use train_models.py to retrain.")

def predict(
    age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate,
    hemoglobin_gdl=None,
    prev_systolic=None, prev_weight=None, curr_weight=None,
    family_hypertension=False, proteinuria=False,
    gravidity=1, parity=0, gestational_age_weeks=20.0,
    bmi_pre_pregnancy=22.0, anemia_status="none",
    hiv_status=False, anc_visits=2,
) -> dict:
    """
    Converts Project 2 vitals → Project 1 feature dict,
    runs both XGBoost models, returns Project-2-compatible result dict.
    All risk level, complications, SHAP and screenings come from the model.
    """
    fasting_glucose_mgdl = round(float(blood_sugar) * 18.0182, 1)

    vitals = {
        "age_years":              age,
        "gravidity":              gravidity,
        "parity":                 parity,
        "gestational_age_weeks":  gestational_age_weeks,
        "bmi_pre_pregnancy":      bmi_pre_pregnancy,
        "systolic_bp_mmhg":       systolic_bp,
        "diastolic_bp_mmhg":      diastolic_bp,
        "hemoglobin_gdl":         hemoglobin_gdl,   # real value now collected
        "fasting_glucose_mgdl":   fasting_glucose_mgdl,
        "proteinuria":            int(proteinuria),
        "hiv_status":             int(hiv_status),
        "anc_visits":             anc_visits,
        "anemia_status":          anemia_status,
    }

    result_p1 = _p1_predict(vitals)

    # Risk level — straight from risk_model
    risk_level = _RISK_TO_UI[result_p1.risk_level]
    risk_score = int(result_p1.risk_prob * 100)

    # Probabilities
    remaining = max(0.0, 1.0 - result_p1.risk_prob)
    rl = result_p1.risk_level
    if rl == "high":
        probs = {"high risk": risk_score, "mid risk": int(remaining*0.65*100), "low risk": int(remaining*0.35*100)}
    elif rl == "moderate":
        probs = {"high risk": int(remaining*0.35*100), "mid risk": risk_score, "low risk": int(remaining*0.65*100)}
    else:
        probs = {"high risk": int(remaining*0.15*100), "mid risk": int(remaining*0.35*100), "low risk": risk_score}

    # Top complications — from complication_model
    top_risks = [c.display_name for c in result_p1.top_conditions
                 if c.condition != "none" and c.probability >= 0.10]
    if not top_risks:
        top_risks = ["No significant complication predicted"]

    # SHAP contributions — from Project 1's TreeExplainer
    shap_reasons = {f["display_name"]: round(f["shap_value"], 4)
                    for f in result_p1.shap_factors[:6]}

    # BP velocity
    bp_velocity = float(systolic_bp - prev_systolic) if prev_systolic is not None else 0.0

    # Suggested action — built from Project 1's CONDITION_SCREENINGS
    action_parts = []
    for s in result_p1.screenings:
        action_parts.append(
            f"[{s.get('urgency','routine').upper()}] {s.get('action','')} — "
            f"{s.get('followup','')} ({s.get('guideline','')})"
        )
    suggested_action = " | ".join(action_parts) if action_parts else "Continue routine antenatal care."

    return {
        "risk_level":       risk_level,
        "risk_score":       risk_score,
        "probabilities":    probs,
        "top_risks":        top_risks,
        "shap_reasons":     shap_reasons,
        "suggested_action": suggested_action,
        "bp_velocity":      bp_velocity,
        "warnings":         result_p1.warnings,
        "explanation_text": result_p1.explanation_text,
        "screenings":       result_p1.screenings,          # full screening dicts
        "top_conditions":   result_p1.top_conditions,      # ComplicationPrediction objects
    }

"""
MaternaAI — Complete Final App
================================
UI     : Project 2 style (fully redesigned)
Models : Project 1 ONLY — XGBoost + SHAP
         risk_level, complications, screenings, SHAP → all from model
         NO rule-based logic anywhere
Run    : streamlit run app.py
"""

import uuid, os, json, subprocess
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import database as db
from model_bridge import predict, load_model
from sms_service import send_post_visit_sms, send_appointment_reminder

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MaternaAI", page_icon="🤰",
                   layout="wide", initial_sidebar_state="expanded")
db.init_db()

@st.cache_resource(show_spinner="Loading AI models…")
def _load():
    return load_model()

try:
    _clf, _le = _load()
    model_ready = True
except Exception:
    model_ready = False

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Colour maps ────────────────────────────────────────────────────────────────
RC  = {"high risk":"#ef4444","mid risk":"#f59e0b","low risk":"#22c55e"}
RBG = {"high risk":"#1a0505","mid risk":"#1a1000","low risk":"#041008"}
RE  = {"high risk":"🔴","mid risk":"🟡","low risk":"🟢"}
UI  = {"critical":"🚨","urgent":"⚡","routine":"📌"}
UC  = {"critical":"#ef4444","urgent":"#f59e0b","routine":"#22c55e"}
UB  = {"critical":"#1a0505","urgent":"#1a1000","routine":"#041008"}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Libre+Baskerville:wght@700&family=JetBrains+Mono:wght@500&display=swap');

:root{
  --bg:#07090e; --s1:#0c1018; --s2:#101620; --s3:#141e2a;
  --b:#1a2d40; --b2:#223650;
  --blue:#2563eb; --cyan:#0891b2; --red:#ef4444;
  --amber:#d97706; --green:#16a34a;
  --text:#dde6f0; --sub:#8899aa; --muted:#4a6070;
  --sans:'Inter',sans-serif; --serif:'Libre Baskerville',Georgia,serif;
  --mono:'JetBrains Mono',monospace;
}

/* Base */
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important; font-family:var(--sans)!important; color:var(--text)!important;
}
#MainMenu,footer,header,[data-testid="stDecoration"]{display:none!important}

/* Sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#090d14 0%,#0c1220 100%)!important;
  border-right:1px solid var(--b)!important;
}
[data-testid="stSidebar"] *{color:var(--text)!important;font-family:var(--sans)!important}
[data-testid="stSidebar"] .stRadio label{
  font-size:0.88rem!important; color:var(--sub)!important;
  padding:6px 10px; border-radius:8px; transition:all 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover{color:var(--text)!important}

/* Typography */
h1,h2,h3{font-family:var(--serif)!important;letter-spacing:-0.01em;color:var(--text)!important}

/* Metrics */
[data-testid="stMetricValue"]{font-family:var(--mono)!important;color:var(--cyan)!important;font-size:1.9rem!important;font-weight:600!important}
[data-testid="stMetricLabel"]{font-size:0.65rem!important;text-transform:uppercase!important;letter-spacing:0.14em!important;color:var(--muted)!important}

/* Primary button */
.stButton>button{
  background:linear-gradient(135deg,#1d4ed8,#0369a1)!important;
  color:#fff!important;border:none!important;border-radius:8px!important;
  font-family:var(--sans)!important;font-weight:600!important;font-size:0.88rem!important;
  padding:0.55rem 1.4rem!important;transition:all 0.2s!important;
}
.stButton>button:hover{opacity:0.88!important;transform:translateY(-1px)!important;box-shadow:0 4px 14px rgba(37,99,235,0.4)!important}

/* Inputs */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div,
.stTextArea>div>div>textarea{
  background:var(--s2)!important;border:1px solid var(--b)!important;
  border-radius:8px!important;color:var(--text)!important;
  font-family:var(--sans)!important;font-size:0.9rem!important;
}
.stTextInput>div>div>input:focus,
.stNumberInput>div>div>input:focus{border-color:var(--blue)!important;box-shadow:0 0 0 3px rgba(37,99,235,0.15)!important}
label{color:var(--sub)!important;font-size:0.8rem!important;font-weight:500!important;letter-spacing:0.02em!important}

/* Tabs */
[data-testid="stTabs"] button{font-weight:600!important;font-size:0.83rem!important;color:var(--muted)!important;letter-spacing:0.03em!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--cyan)!important;border-bottom-color:var(--cyan)!important}

/* Expander */
[data-testid="stExpander"]{background:var(--s1)!important;border:1px solid var(--b)!important;border-radius:10px!important}

/* Progress */
.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--blue),var(--cyan))!important}

/* Checkbox */
.stCheckbox label{color:var(--sub)!important;font-size:0.88rem!important}

/* Divider */
hr{border-color:var(--b)!important}

/* Alerts */
[data-testid="stAlert"]{border-radius:10px!important;border-left-width:4px!important;font-size:0.88rem!important}

/* ── Custom components ── */
.section-title{
  font-family:var(--serif);font-size:1.5rem;color:var(--text);
  border-bottom:1px solid var(--b);padding-bottom:10px;margin-bottom:1.4rem;
  display:flex;align-items:center;gap:10px;
}

.kpi-card{
  background:var(--s1);border:1px solid var(--b);
  border-left:4px solid;border-radius:12px;
  padding:1.1rem 1.4rem;margin-bottom:4px;
}
.kpi-label{font-size:10px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:4px}
.kpi-val{font-family:var(--mono);font-size:1.9rem;font-weight:700;color:var(--text);line-height:1}
.kpi-sub{font-size:11px;color:var(--muted);margin-top:4px}

.badge{
  display:inline-flex;align-items:center;gap:6px;
  padding:5px 16px;border-radius:999px;font-weight:700;
  font-size:12px;letter-spacing:0.06em;text-transform:uppercase;border:2px solid;
}

.pcard{
  background:var(--s1);border:1px solid var(--b);border-radius:12px;
  padding:12px 16px;margin-bottom:8px;
  display:flex;align-items:center;gap:12px;transition:border-color 0.2s;
}
.pcard:hover{border-color:var(--cyan)}

/* Form section box */
.form-section{
  background:var(--s1);border:1px solid var(--b);border-radius:12px;
  padding:1.2rem 1.4rem;margin-bottom:1rem;
}
.form-section-title{
  font-size:0.78rem;font-weight:700;text-transform:uppercase;
  letter-spacing:0.12em;color:var(--cyan);margin-bottom:1rem;
  display:flex;align-items:center;gap:8px;
}

/* Screening card */
.sc{border-left:4px solid;border-radius:10px;padding:14px 18px;margin-bottom:10px}
.sc-critical{background:#1a0505;border-color:#ef4444;color:#fca5a5}
.sc-urgent  {background:#1a1000;border-color:#d97706;color:#fcd34d}
.sc-routine {background:#041008;border-color:#16a34a;color:#86efac}

/* Risk banner */
.risk-banner{
  border:2px solid;border-radius:14px;padding:1.2rem 1.6rem;
  margin-bottom:1.2rem;display:flex;align-items:center;gap:16px;
}

/* Alert animated */
.rt-crit{
  background:linear-gradient(135deg,#7f1d1d,#991b1b);color:#fff;
  border-radius:10px;padding:14px 18px;font-weight:700;font-size:0.9rem;
  border:1px solid #ef4444;animation:pulse 1.8s ease-in-out infinite;
}
.rt-warn{
  background:#1a1000;color:#fcd34d;border-radius:10px;
  padding:12px 16px;font-size:0.88rem;border:1px solid #d97706;
}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.75}}

/* Model status pill */
.model-pill{
  display:inline-flex;align-items:center;gap:8px;
  border-radius:999px;padding:6px 16px;font-size:0.75rem;font-weight:700;
}
.model-ok{background:#041008;border:1px solid #16a34a;color:#22c55e}
.model-err{background:#1a0505;border:1px solid #ef4444;color:#ef4444}

/* Sidebar nav pill */
.nav-pill{
  display:block;padding:10px 14px;border-radius:10px;margin-bottom:4px;
  font-size:0.88rem;font-weight:500;cursor:pointer;transition:all 0.15s;
  border:1px solid transparent;color:var(--sub);
}
.nav-pill:hover{background:var(--s2);color:var(--text);border-color:var(--b)}
.nav-pill.active{background:var(--s3);color:var(--cyan);border-color:var(--b2);font-weight:600}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown("""
    <div style="padding:1.2rem 0.4rem 0.6rem">
      <div style="font-family:'Libre Baskerville',serif;font-size:1.6rem;
           color:#dde6f0;letter-spacing:-0.02em;line-height:1">
        Materna<span style="color:#0891b2">AI</span>
      </div>
      <div style="font-size:0.68rem;color:#4a6070;text-transform:uppercase;
           letter-spacing:0.14em;margin-top:4px">Maternal Risk System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Model status
    if model_ready:
        st.markdown('<div class="model-pill model-ok">✓ AI Models Ready · XGBoost + SHAP</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-pill model-err">✗ Models not loaded — go to Settings</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Dashboard",
        "🆕  Register Patient",
        "💉  Today's Visit",
        "👩  Patient Records",
        "📅  Appointments",
        "📱  SMS Log",
        "🧠  Train Models",
        "⚙️  Settings",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("""
    <div style="font-size:10px;color:#2d3f50;line-height:1.9;padding:0 0.2rem">
      Kenyan Maternal Health<br>
      XGBoost · SHAP · SQLite<br>
      SMS via Africa's Talking
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
def kpi(label, value, sub="", colour="#2563eb"):
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color:{colour}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-val">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def risk_badge(level):
    c=RC.get(level,"#64748b"); b=RBG.get(level,"#111"); e=RE.get(level,"⚪")
    st.markdown(f'<span class="badge" style="background:{b};color:{c};border-color:{c}">{e} {level.upper()}</span>', unsafe_allow_html=True)

def shap_chart(shap_reasons):
    if not shap_reasons: return
    labels = list(shap_reasons.keys())
    values = list(shap_reasons.values())
    colours = ["#ef4444" if v>0 else "#22c55e" for v in values]
    fig = go.Figure(go.Bar(x=values,y=labels,orientation="h",
        marker_color=colours,text=[f"{v:+.3f}" for v in values],textposition="outside"))
    fig.update_layout(
        title="🔍 SHAP Feature Contributions — What drove this prediction",
        xaxis_title="Impact on risk  (red = raises risk · green = lowers risk)",
        height=max(240,len(labels)*44),
        margin=dict(l=10,r=70,t=50,b=10),
        plot_bgcolor="#0c1018",paper_bgcolor="#0c1018",
        font=dict(color="#8899aa",size=11,family="Inter"),
        title_font=dict(color="#dde6f0",size=13),
        xaxis=dict(zeroline=True,zerolinecolor="#1a2d40",gridcolor="#101620",color="#64748b"),
        yaxis=dict(gridcolor="#101620",color="#8899aa"),
    )
    st.plotly_chart(fig, use_container_width=True)

def render_screenings(screenings):
    """Render Project 1's condition-specific screening recommendations."""
    CNAMES = {
        "none":"No Complication","gestational_diabetes":"Gestational Diabetes",
        "preeclampsia":"Preeclampsia","severe_anemia":"Severe Anaemia",
        "hemorrhage":"Haemorrhage","eclampsia":"Eclampsia",
    }
    for s in screenings:
        u   = s.get("urgency","routine")
        css = f"sc sc-{u}"
        cond= CNAMES.get(s.get("condition",""),s.get("condition",""))
        st.markdown(f"""
        <div class="{css}">
          <div style="font-weight:700;font-size:0.92rem;margin-bottom:6px">{UI.get(u,"📌")} {cond}</div>
          <div style="font-size:0.88rem;margin-bottom:4px"><b>Action:</b> {s.get('action','')}</div>
          <div style="font-size:0.85rem;margin-bottom:4px"><b>Follow-up:</b> {s.get('followup','')}</div>
          <div style="font-size:0.72rem;opacity:0.65">📖 {s.get('guideline','')}</div>
        </div>""", unsafe_allow_html=True)

def trend_chart(visits, name):
    if len(visits)<2:
        st.info("Need ≥2 visits for trend charts.")
        return
    df=pd.DataFrame(visits).sort_values("visit_date")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["visit_date"],y=df["systolic_bp"],
        name="Systolic BP",line=dict(color="#ef4444",width=2),mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df["visit_date"],y=df["diastolic_bp"],
        name="Diastolic BP",line=dict(color="#f59e0b",width=2),mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df["visit_date"],y=df["blood_sugar"],
        name="Blood Sugar (mmol/L)",line=dict(color="#0891b2",width=2),
        mode="lines+markers",yaxis="y2"))
    fig.add_hrect(y0=90,y1=135,fillcolor="#22c55e",opacity=0.04,line_width=0)
    fig.update_layout(
        title=f"📈 {name} — Vital Trends",
        xaxis_title="Visit Date",yaxis_title="Blood Pressure (mmHg)",
        yaxis2=dict(title="Blood Sugar (mmol/L)",overlaying="y",side="right",color="#0891b2"),
        height=330,plot_bgcolor="#0c1018",paper_bgcolor="#0c1018",
        font=dict(color="#8899aa",family="Inter"),title_font=dict(color="#dde6f0"),
        legend=dict(orientation="h",y=-0.28,bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#101620",color="#64748b"),
        yaxis=dict(gridcolor="#101620",color="#64748b"),
        margin=dict(l=10,r=10,t=46,b=40),
    )
    st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    st.markdown('<div class="section-title">📊 Clinical Overview</div>', unsafe_allow_html=True)

    patients = db.list_patients()
    with db.get_conn() as conn:
        hr_count = conn.execute(
            """SELECT COUNT(DISTINCT patient_id) FROM predictions
               WHERE pred_id IN (SELECT MAX(pred_id) FROM predictions GROUP BY patient_id)
               AND risk_level='high risk'""").fetchone()[0]
        tv = conn.execute("SELECT COUNT(*) FROM visits WHERE visit_date=date('now')").fetchone()[0]
        pa = conn.execute("SELECT COUNT(*) FROM appointments WHERE status='scheduled' AND scheduled_date>=date('now')").fetchone()[0]

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Total Patients",   len(patients), "Registered",             "#2563eb")
    with c2: kpi("🔴 High Risk",     hr_count,      "Require urgent review",  "#ef4444")
    with c3: kpi("🏥 Visits Today",  tv,            str(date.today()),        "#0891b2")
    with c4: kpi("📅 Appointments",  pa,            "Upcoming scheduled",     "#16a34a")

    st.markdown("---")
    cl,cr = st.columns([1.5,1])

    with cl:
        st.markdown("#### 🚨 High-Priority Patients")
        high_risk = db.get_high_risk_patients()
        if not high_risk:
            st.success("✅ No high-risk patients flagged right now.")
        else:
            for p in high_risk[:8]:
                score=int(p.get("risk_score",0))
                st.markdown(f"""
                <div class="pcard">
                  <div style="width:38px;height:38px;border-radius:50%;background:#1a0505;
                       display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0">🔴</div>
                  <div style="flex:1;min-width:0">
                    <div style="font-weight:600;color:#dde6f0;font-size:0.92rem">{p['full_name']}</div>
                    <div style="font-size:11px;color:#4a6070">ID: {p['patient_id']} · Risk score: {score}%</div>
                    <div style="font-size:11px;color:#374151;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{str(p.get('suggested_action',''))[:85]}…</div>
                  </div>
                  <span style="background:#1a0505;color:#ef4444;border:1.5px solid #ef4444;
                        border-radius:99px;padding:3px 10px;font-size:11px;font-weight:700;flex-shrink:0">{score}%</span>
                </div>""", unsafe_allow_html=True)

    with cr:
        st.markdown("#### 📊 Risk Distribution")
        with db.get_conn() as conn:
            dist = conn.execute(
                """SELECT risk_level, COUNT(*) as cnt FROM predictions
                   WHERE pred_id IN (SELECT MAX(pred_id) FROM predictions GROUP BY patient_id)
                   GROUP BY risk_level""").fetchall()
        if dist:
            df_d=pd.DataFrame([dict(r) for r in dist])
            fig=px.pie(df_d,names="risk_level",values="cnt",color="risk_level",
                color_discrete_map={"high risk":"#ef4444","mid risk":"#f59e0b","low risk":"#22c55e"},hole=0.6)
            fig.update_traces(textposition="outside",textinfo="percent+label",textfont=dict(color="#8899aa",size=11))
            fig.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),showlegend=False,paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("No predictions yet — run a visit first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGISTER PATIENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🆕  Register Patient":
    st.markdown('<div class="section-title">🆕 Register New Patient</div>', unsafe_allow_html=True)
    st.caption("All fields are stored permanently. Patient ID is auto-generated as **XXXXXX/YYYY**.")

    # Show last registered ID if available
    if st.session_state.get("last_registered_id"):
        lr = st.session_state["last_registered_id"]
        st.markdown(f"""
        <div style="background:#041008;border:1px solid #16a34a;border-radius:10px;
             padding:12px 18px;margin-bottom:1rem;display:flex;align-items:center;gap:12px">
          <div style="font-size:1.4rem">✅</div>
          <div>
            <div style="color:#22c55e;font-weight:700;font-size:0.9rem">Last registered patient</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                 color:#dde6f0;font-weight:700;letter-spacing:0.1em">{lr['pid']}</div>
            <div style="color:#4a6070;font-size:0.8rem">{lr['name']}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with st.form("register_form", clear_on_submit=False):

        # ── Section 1: Personal Information ──────────────────────────────────
        st.markdown('<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#0891b2;margin-bottom:0.8rem">👤 Personal Information</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            full_name = st.text_input("Full Name ✱")
            phone     = st.text_input("Phone Number ✱", placeholder="+254712345678")
        with col2:
            age       = st.number_input("Age (years) ✱", 14, 55, 25)
            lmp       = st.date_input("Last Menstrual Period (LMP) ✱",
                                       value=date.today()-timedelta(weeks=20),
                                       help="Used to auto-calculate gestational age at each visit")
        with col3:
            fam_htn   = st.checkbox("Family History of Hypertension")
            hiv       = st.checkbox("HIV Positive")

        st.divider()

        # ── Section 2: Obstetric Profile ─────────────────────────────────────
        st.markdown('<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#0891b2;margin-bottom:0.8rem">🤰 Obstetric Profile</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            gravidity = st.number_input("Gravidity (total pregnancies incl. current) ✱", 0, 20, 1)
            parity    = st.number_input("Parity (births ≥ 24 weeks) ✱", 0, 20, 0)
        with col5:
            bmi       = st.number_input("Pre-pregnancy BMI ✱", 10.0, 60.0, 22.0, 0.1)
            anc_prior = st.number_input("Prior ANC Visits (before this facility)", 0, 20, 0,
                                         help="Visits at other facilities before registering here. Subsequent visits here are counted automatically.")
        with col6:
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            st.caption("ℹ️ **Gestational age** is auto-calculated from LMP at each visit.\n\n**ANC count** = prior visits entered here + visits saved in this system.")

        st.divider()
        submitted = st.form_submit_button("💾 Register Patient", use_container_width=True, type="primary")

    if submitted:
        if not full_name or not phone:
            st.error("❌ Full name and phone number are required.")
        else:
            # ID: 6 numeric digits / last 2 digits of year  e.g. 483920/25
            import random
            serial   = str(random.randint(100000, 999999))
            yr2      = str(date.today().year)[-2:]
            pid      = f"{serial}/{yr2}"

            reg_gest = max(0.0, min(42.0, round((date.today() - lmp).days / 7.0, 1)))

            db.upsert_patient(
                pid, full_name, phone, age,
                int(reg_gest), fam_htn, hiv,
                gravidity, parity, bmi,
                lmp_date=lmp.isoformat(),
                anc_prior=int(anc_prior),
            )
            st.session_state["last_registered_id"] = {"pid": pid, "name": full_name}
            st.success("✅ Patient registered!")
            st.markdown(f"""
            <div style="background:#0c1018;border:2px solid #2563eb;border-radius:12px;
                 padding:1.2rem 1.6rem;margin-top:0.6rem">
              <div style="font-size:0.72rem;color:#4a6070;text-transform:uppercase;
                   letter-spacing:0.12em;margin-bottom:6px">Patient ID — use this for search</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;
                   font-weight:700;color:#dde6f0;letter-spacing:0.1em">{pid}</div>
              <div style="font-size:0.82rem;color:#8899aa;margin-top:6px">
                {full_name} · Age {age} · LMP: {lmp} · Gest. age: {reg_gest:.1f} wks · Prior ANC: {anc_prior}
              </div>
            </div>""", unsafe_allow_html=True)
            st.info("Go to **💉 Today's Visit** to record the first assessment.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TODAY'S VISIT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💉  Today's Visit":
    st.markdown('<div class="section-title">💉 Today\'s Visit & AI Assessment</div>', unsafe_allow_html=True)

    patients = db.list_patients()
    if not patients:
        st.warning("No patients registered yet. Go to **🆕 Register Patient** first.")
        st.stop()

    # Search by ID or name
    search_val = st.text_input("🔍 Search patient by ID (e.g. A3F9K2/2025) or name", "")
    if search_val:
        filtered_p = [p for p in patients
                      if search_val.upper() in p["patient_id"].upper()
                      or search_val.lower() in p["full_name"].lower()]
    else:
        filtered_p = patients

    if not filtered_p:
        st.warning("No patient found. Check the ID or name.")
        st.stop()

    p_opts = {f"{p['full_name']}  ·  {p['patient_id']}": p for p in filtered_p}
    sel    = st.selectbox("Select Patient", list(p_opts.keys()))
    pat    = p_opts[sel]
    prev_visits = db.get_patient_visits(pat["patient_id"])

    # ── Auto-calculate gestational age from LMP ───────────────────────────────
    lmp_str  = pat.get("lmp_date")
    if lmp_str:
        lmp_date   = date.fromisoformat(lmp_str)
        auto_gest  = round((date.today() - lmp_date).days / 7.0, 1)
        auto_gest  = max(0.0, min(42.0, auto_gest))
    else:
        auto_gest  = float(pat.get("weeks_pregnant", 20))

    # ── Auto-calculate ANC = prior visits at registration + visits saved here ──
    auto_anc = int(pat.get("anc_prior", 0)) + len(prev_visits)

    # ── Show patient summary strip ────────────────────────────────────────────
    gest_display = f"{auto_gest:.1f}" if lmp_str else f"~{auto_gest:.0f}"
    st.markdown(f"""
    <div style="background:#0c1018;border:1px solid #1a2d40;border-radius:10px;
         padding:10px 18px;margin-bottom:1rem;display:flex;gap:24px;align-items:center;flex-wrap:wrap">
      <div>
        <div style="font-size:10px;color:#4a6070;text-transform:uppercase;letter-spacing:0.1em">Patient</div>
        <div style="font-weight:700;color:#dde6f0">{pat['full_name']}</div>
      </div>
      <div>
        <div style="font-size:10px;color:#4a6070;text-transform:uppercase;letter-spacing:0.1em">ID</div>
        <div style="font-family:'JetBrains Mono',monospace;color:#0891b2;font-weight:700">{pat['patient_id']}</div>
      </div>
      <div>
        <div style="font-size:10px;color:#4a6070;text-transform:uppercase;letter-spacing:0.1em">Gestational Age (today)</div>
        <div style="font-weight:700;color:#dde6f0">{gest_display} weeks</div>
      </div>
      <div>
        <div style="font-size:10px;color:#4a6070;text-transform:uppercase;letter-spacing:0.1em">ANC Visits (auto)</div>
        <div style="font-weight:700;color:#dde6f0">{auto_anc} visit{'s' if auto_anc!=1 else ''}</div>
      </div>
      <div>
        <div style="font-size:10px;color:#4a6070;text-transform:uppercase;letter-spacing:0.1em">HIV</div>
        <div style="font-weight:700;color:{'#ef4444' if pat.get('hiv_status') else '#22c55e'}">{'Positive' if pat.get('hiv_status') else 'Negative'}</div>
      </div>
      <div>
        <div style="font-size:10px;color:#4a6070;text-transform:uppercase;letter-spacing:0.1em">Today</div>
        <div style="font-weight:700;color:#dde6f0">{date.today().strftime('%d %b %Y')}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Today's Vitals (Section B only) ──────────────────────────────────────
    st.markdown('<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#0891b2;margin-bottom:0.8rem">💉 Today\'s Vitals</div>', unsafe_allow_html=True)

    cb1, cb2, cb3 = st.columns(3)
    with cb1:
        systolic    = st.number_input("Systolic BP (mmHg)",  60, 220, 120)
        diastolic   = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)
        hb          = st.number_input("Haemoglobin (g/dL)",  3.0, 20.0, 11.5, 0.1,
                        help="Normal in pregnancy: ≥11 g/dL | Severe anaemia: <7 g/dL")
    with cb2:
        glucose     = st.number_input("Blood Sugar (mmol/L)", 3.0, 20.0, 7.5, 0.1)
        body_temp   = st.number_input("Body Temperature (°F)", 95.0, 105.0, 98.2, 0.1)
        anemia_st   = st.selectbox("Anemia Status",
                        ["none","mild","moderate","severe"],
                        help="Current anemia classification based on Hb and symptoms")
    with cb3:
        heart_rate  = st.number_input("Heart Rate (bpm)", 40, 160, 76)
        weight_kg   = st.number_input("Weight (kg)",      35.0, 150.0, 65.0, 0.5)
        proteinuria = st.checkbox("Proteinuria detected (dipstick +)")

    notes = st.text_area("Clinical Notes", placeholder="Any observations or remarks…", height=68)

    # Visual-only alerts (display only — do NOT feed into model)
    if systolic >= 160 or diastolic >= 110:
        st.markdown(f'<div class="rt-crit">🚨 BP {systolic}/{diastolic} mmHg — Critical reading, seek physician immediately</div>', unsafe_allow_html=True)
    elif systolic >= 140 or diastolic >= 90:
        st.markdown(f'<div class="rt-warn">⚠️ BP {systolic}/{diastolic} mmHg — Elevated, document and monitor</div>', unsafe_allow_html=True)
    if hb < 7.0:
        st.markdown('<div class="rt-crit">🚨 Haemoglobin critically low — Severe anaemia, urgent intervention needed</div>', unsafe_allow_html=True)
    elif hb < 11.0:
        st.markdown(f'<div class="rt-warn">⚠️ Haemoglobin {hb} g/dL — Below pregnancy threshold (11 g/dL)</div>', unsafe_allow_html=True)

    st.divider()

    if st.button("🧠 Run AI Assessment & Save Visit", use_container_width=True, type="primary"):
        if not model_ready:
            st.error("❌ AI models not loaded. Go to 🧠 Train Models to prepare them.")
        else:
            prev_s = prev_visits[0]["systolic_bp"] if prev_visits else None
            prev_w = prev_visits[0]["weight_kg"]   if prev_visits else None

            visit_id = db.save_visit(pat["patient_id"], systolic, diastolic, glucose,
                                      body_temp, heart_rate, weight_kg, proteinuria, notes)

            with st.spinner("🧠 Running XGBoost + SHAP analysis…"):
                result = predict(
                    age=pat["age"],
                    systolic_bp=systolic, diastolic_bp=diastolic,
                    blood_sugar=glucose,  body_temp=body_temp,
                    heart_rate=heart_rate, hemoglobin_gdl=hb,
                    prev_systolic=prev_s,  prev_weight=prev_w,
                    curr_weight=weight_kg,
                    family_hypertension=bool(pat.get("family_hypertension")),
                    proteinuria=proteinuria,
                    gravidity=int(pat.get("gravidity", 1)),
                    parity=int(pat.get("parity", 0)),
                    gestational_age_weeks=auto_gest,          # auto from LMP
                    bmi_pre_pregnancy=float(pat.get("bmi_pre_pregnancy", 22.0)),
                    anemia_status=anemia_st,                  # entered today
                    hiv_status=bool(pat.get("hiv_status", False)),
                    anc_visits=auto_anc,                      # auto from visit count
                )

            db.save_prediction(visit_id, pat["patient_id"],
                result["risk_level"], result["risk_score"],
                result["top_risks"], result["shap_reasons"],
                result["suggested_action"])

            days_ahead = 3 if result["risk_level"]=="high risk" else 14 if result["risk_level"]=="mid risk" else 28
            next_date  = (date.today()+timedelta(days=days_ahead)).isoformat()
            db.schedule_appointment(pat["patient_id"], next_date, f"Follow-up: {result['risk_level']}")
            if pat.get("phone"):
                send_post_visit_sms(pat["patient_id"], pat["full_name"], pat["phone"], next_date)

            # ── RESULTS ────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🤖 AI Assessment Results")
            st.caption("All results — risk level, complications, SHAP, recommendations — come exclusively from the trained XGBoost models.")

            # Risk banner
            rl=result["risk_level"]; rc=RC.get(rl,"#64748b"); bg=RBG.get(rl,"#111")
            st.markdown(f"""
            <div class="risk-banner" style="background:{bg};border-color:{rc}">
              <div style="font-size:2.4rem">{RE.get(rl,'⚪')}</div>
              <div>
                <div style="font-family:'Libre Baskerville',serif;font-size:1.35rem;
                     color:{rc};font-weight:700;letter-spacing:-0.01em">{rl.upper()}</div>
                <div style="color:#8899aa;font-size:0.82rem;margin-top:3px">
                  Model confidence: <b style="color:{rc}">{result['risk_score']}%</b>
                  &nbsp;·&nbsp; Gest. age used: <b style="color:#dde6f0">{auto_gest:.1f} wks</b>
                  &nbsp;·&nbsp; ANC visits used: <b style="color:#dde6f0">{auto_anc}</b>
                  &nbsp;·&nbsp; Next appt: <b style="color:#dde6f0">{next_date}</b>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Complications — full width
            st.markdown("**🔬 Predicted Complications** *(from complication model)*")
            if result.get("bp_velocity",0)!=0:
                d="↑" if result["bp_velocity"]>0 else "↓"
                st.markdown(f'<div style="font-size:11px;color:#8899aa;margin-bottom:8px">📈 BP {d} {abs(result["bp_velocity"]):.0f} mmHg since last visit</div>', unsafe_allow_html=True)
            for c in result.get("top_conditions",[]):
                bar_col="#ef4444" if c.rank==1 else "#f59e0b" if c.rank==2 else "#22c55e"
                medal="🥇" if c.rank==1 else "🥈" if c.rank==2 else "🥉"
                st.markdown(f"""
                <div style="margin-bottom:12px">
                  <div style="display:flex;justify-content:space-between;font-size:12px;
                       color:#8899aa;margin-bottom:4px">
                    <span>{medal} {c.display_name}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-weight:700;
                         color:{bar_col}">{c.probability:.0%}</span>
                  </div>
                  <div style="background:#1a2d40;border-radius:99px;height:10px">
                    <div style="background:{bar_col};width:{min(int(c.probability*100),100)}%;
                         height:10px;border-radius:99px;transition:width 0.4s"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # SHAP
            st.markdown("---")
            shap_chart(result["shap_reasons"])

            if result.get("explanation_text"):
                st.markdown("**🗣️ Plain-language explanation**")
                for line in result["explanation_text"]:
                    st.markdown(f'<div style="font-size:0.87rem;color:#8899aa;padding:2px 0 2px 10px;border-left:3px solid #1a2d40">→ {line}</div>', unsafe_allow_html=True)

            # Screenings from Project 1's CONDITION_SCREENINGS
            st.markdown("---")
            st.markdown("**📋 Clinical Screening Recommendations** *(generated by AI model)*")
            st.caption("Based on predicted conditions from the complication model. Physician review recommended.")
            render_screenings(result.get("screenings",[]))

            if result.get("warnings"):
                with st.expander("⚠️ Data Notes"):
                    for w in result["warnings"]:
                        st.caption(f"• {w}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT RECORDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👩  Patient Records":
    st.markdown('<div class="section-title">👩 Patient Records</div>', unsafe_allow_html=True)

    patients=db.list_patients()
    if not patients:
        st.info("No patients registered yet.")
        st.stop()

    search=st.text_input("🔍 Search by name or ID","")
    filtered=[p for p in patients if search.lower() in p["full_name"].lower()
              or search.lower() in p["patient_id"].lower()] if search else patients

    if not filtered:
        st.warning("No patients match your search.")
        st.stop()

    sel_l=st.selectbox("Select patient",[f"{p['full_name']} ({p['patient_id']})" for p in filtered])
    pid=sel_l.split("(")[-1].rstrip(")")
    pat=db.get_patient(pid)
    pred=db.get_latest_prediction(pid)
    visits=db.get_patient_visits(pid)

    col1,col2=st.columns([1,2])
    with col1:
        fam="⚠️ Family hypertension" if pat.get("family_hypertension") else "No family hypertension"
        st.markdown(f"""
        <div class="kpi-card card-accent" style="border-left-color:#2563eb">
          <div style="font-size:2.2rem;margin-bottom:8px">🤰</div>
          <div style="font-size:1.05rem;font-weight:700;color:#dde6f0">{pat['full_name']}</div>
          <div style="color:#4a6070;font-size:11px;margin-bottom:10px">ID: {pat['patient_id']}</div>
          <div style="font-size:12px;line-height:2.1;color:#8899aa">
            📱 {pat.get('phone','—')}<br>
            🗓️ Age: {pat.get('age','—')} yrs<br>
            🤱 {pat.get('weeks_pregnant','—')} weeks pregnant<br>
            👶 {pat.get('prev_pregnancies','—')} prior pregnancy(ies)<br>
            🏥 {fam}
          </div>
        </div>""", unsafe_allow_html=True)
        if pred:
            st.markdown("**Latest Assessment**")
            risk_badge(pred["risk_level"])
            st.metric("Risk Score",f"{pred['risk_score']}%")

    with col2:
        if visits:
            trend_chart(visits, pat["full_name"])
            with st.expander("📋 Full Visit History"):
                df_v=pd.DataFrame(visits)[["visit_date","systolic_bp","diastolic_bp","blood_sugar","heart_rate","weight_kg"]]
                df_v.columns=["Date","Systolic","Diastolic","Blood Sugar","HR","Weight kg"]
                st.dataframe(df_v,use_container_width=True,hide_index=True)
        else:
            st.info("No visit history yet.")

    if pred:
        st.markdown("---")
        s1,s2=st.columns(2)
        with s1:
            shap_chart(pred["shap_reasons"])
        with s2:
            st.markdown("**🔬 Predicted Conditions**")
            for cond in pred.get("top_risks",[]):
                st.markdown(f'<div style="font-size:0.88rem;color:#8899aa;padding:2px 0">• {cond}</div>', unsafe_allow_html=True)
            st.markdown("<br>**📋 Model Recommendation**", unsafe_allow_html=True)
            rl=pred.get("risk_level","low risk")
            css="sc sc-critical" if rl=="high risk" else "sc sc-urgent" if rl=="mid risk" else "sc sc-routine"
            st.markdown(f'<div class="{css}" style="font-size:0.85rem">{pred.get("suggested_action","")}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: APPOINTMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅  Appointments":
    st.markdown('<div class="section-title">📅 Appointments</div>', unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["📌 Upcoming","✅ Mark Attendance","➕ Book"])

    with t1:
        upcoming=db.get_upcoming_appointments(days_ahead=14)
        if not upcoming:
            st.info("No upcoming appointments in the next 14 days.")
        else:
            for a in upcoming:
                dl=(pd.to_datetime(a["scheduled_date"])-pd.Timestamp.today()).days
                colour="#ef4444" if dl<=1 else "#f59e0b" if dl<=3 else "#22c55e"
                st.markdown(f"""
                <div class="pcard">
                  <div style="width:36px;height:36px;border-radius:50%;background:{colour}20;
                       display:flex;align-items:center;justify-content:center;font-size:16px">📅</div>
                  <div style="flex:1">
                    <div style="font-weight:600;color:#dde6f0;font-size:0.9rem">{a['full_name']}</div>
                    <div style="font-size:11px;color:#4a6070">{a['scheduled_date']} · {a.get('reason','')}</div>
                  </div>
                  <span style="background:{colour}20;color:{colour};border:1.5px solid {colour};
                        border-radius:99px;padding:3px 10px;font-size:11px;font-weight:700">{max(0,dl)}d</span>
                </div>""", unsafe_allow_html=True)

    with t2:
        with db.get_conn() as conn:
            pending=conn.execute(
                """SELECT a.appt_id,a.scheduled_date,a.reason,p.full_name,p.patient_id,p.phone
                   FROM appointments a JOIN patients p ON a.patient_id=p.patient_id
                   WHERE a.status='scheduled' ORDER BY a.scheduled_date""").fetchall()
        if not pending:
            st.info("No pending appointments.")
        else:
            opts={f"{r['full_name']} — {r['scheduled_date']} (#{r['appt_id']})":r for r in pending}
            sel=st.selectbox("Select",list(opts.keys()))
            ap=opts[sel]
            b1,b2=st.columns(2)
            with b1:
                if st.button("✅ Mark Attended",use_container_width=True):
                    db.mark_appointment_attended(ap["appt_id"]); st.success("Marked attended!"); st.rerun()
            with b2:
                if st.button("❌ Mark Missed",use_container_width=True):
                    db.mark_appointment_missed(ap["appt_id"])
                    p2=db.get_patient(ap["patient_id"])
                    if p2 and p2.get("phone"):
                        from sms_service import send_missed_appointment_sms
                        send_missed_appointment_sms(ap["patient_id"],ap["full_name"],p2["phone"])
                    st.warning("Marked missed. SMS sent."); st.rerun()

    with t3:
        pts=db.list_patients()
        if not pts:
            st.info("Register a patient first.")
        else:
            with st.form("book"):
                po={f"{p['full_name']} ({p['patient_id']})":p for p in pts}
                sp=st.selectbox("Patient",list(po.keys()))
                ad=st.date_input("Date",value=date.today()+timedelta(days=7))
                rs=st.text_input("Reason","Antenatal follow-up")
                if st.form_submit_button("📅 Book & Send Reminder",use_container_width=True):
                    p3=po[sp]; db.schedule_appointment(p3["patient_id"],str(ad),rs)
                    if p3.get("phone"): send_appointment_reminder(p3["patient_id"],p3["full_name"],p3["phone"],str(ad))
                    st.success(f"✅ Booked for {ad}. SMS sent.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SMS LOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📱  SMS Log":
    st.markdown('<div class="section-title">📱 SMS Log</div>', unsafe_allow_html=True)
    logs=db.get_sms_log()
    if not logs:
        st.info("No messages sent yet.")
    else:
        df=pd.DataFrame(logs)
        df["status_icon"]=df["status"].map({"sent":"✅","mock_success":"🔵","failed":"❌","queued":"⏳"}).fillna("❓")
        st.dataframe(df[["sent_at","patient_id","phone","message_type","status_icon","message_body"]],
            use_container_width=True,hide_index=True,
            column_config={"sent_at":"Time","patient_id":"Patient ID","phone":"Phone",
                "message_type":"Type","status_icon":"Status",
                "message_body":st.column_config.TextColumn("Message",width="large")})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Train Models":
    st.markdown('<div class="section-title">🧠 Train AI Models</div>', unsafe_allow_html=True)

    # Model status cards
    mc1,mc2=st.columns(2)
    with mc1:
        st.markdown("""
        <div class="kpi-card" style="border-left-color:#2563eb">
          <div class="kpi-label">Complication Model</div>
          <div style="color:#dde6f0;font-weight:600;margin-bottom:4px">XGBoost · 6 classes</div>
          <div style="color:#4a6070;font-size:12px;line-height:1.7">
            None · Gestational Diabetes · Preeclampsia<br>Severe Anaemia · Haemorrhage · Eclampsia
          </div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div class="kpi-card" style="border-left-color:#0891b2">
          <div class="kpi-label">Risk Level Model</div>
          <div style="color:#dde6f0;font-weight:600;margin-bottom:4px">XGBoost · 3 classes</div>
          <div style="color:#4a6070;font-size:12px;line-height:1.7">
            Low · Moderate · High<br>SHAP TreeExplainer · 13 features
          </div>
        </div>""", unsafe_allow_html=True)

    # Show metrics if available
    mpath=os.path.join(BASE,"model_metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            mx=json.load(f)
        st.markdown("#### 📊 Current Model Performance")
        m1,m2,m3,m4=st.columns(4)
        cm=mx.get("complication_model",{}); rm=mx.get("risk_model",{})
        m1.metric("Complication Accuracy", f"{cm.get('accuracy',0):.1%}")
        m2.metric("Complication Macro F1",  f"{cm.get('macro_f1',0):.1%}")
        m3.metric("Risk Accuracy",          f"{rm.get('accuracy',0):.1%}")
        m4.metric("Risk Macro F1",          f"{rm.get('macro_f1',0):.1%}")

    if model_ready:
        st.success("✅ Models are loaded and ready.")
    else:
        st.error("❌ Models not found — run Steps 1 & 2 below.")

    st.markdown("---")

    # Step 1
    st.markdown("### 📥 Step 1 — Prepare Training Data")
    st.markdown("""
    Downloads the **electricsheepafrica/maternal-health-pregnancy** dataset from HuggingFace,
    cleans it, encodes features, applies **SMOTE** to balance rare conditions (eclampsia, haemorrhage),
    then saves `train_complication.csv`, `train_risk.csv`, `test.csv`, `encoders.pkl`, `feature_names.pkl`.
    """)
    st.caption("⚠️ Requires internet connection. Takes ~1–2 minutes.")
    if st.button("📥 Run prepare_data.py", use_container_width=True):
        with st.spinner("Downloading dataset and preparing data…"):
            r=subprocess.run(["python", os.path.join(BASE,"prepare_data.py")],
                capture_output=True,text=True,cwd=BASE)
        if r.returncode==0:
            st.success("✅ Data prepared successfully!")
            with st.expander("📋 Output log"):
                st.code(r.stdout[-3000:] if len(r.stdout)>3000 else r.stdout or "(no output)")
        else:
            st.error("❌ prepare_data.py failed:")
            st.code(r.stderr[-2000:])

    st.markdown("---")

    # Step 2
    st.markdown("### 🚀 Step 2 — Train Both Models")
    st.markdown("""
    Trains **Model 1** (complication predictor, 6 classes) and **Model 2** (risk level, 3 classes)
    using XGBoost with per-sample class weights. Evaluates on the held-out test set and prints
    classification reports with **Recall**, **Precision**, **F1** per class.
    Saves `complication_model.pkl`, `risk_model.pkl`, `model_metrics.json`.
    """)
    st.caption("⚠️ Requires Step 1 to have been run first. Takes ~30–90 seconds.")
    if st.button("🚀 Run train_models.py", use_container_width=True, type="primary"):
        with st.spinner("Training XGBoost models… (30–90 seconds)"):
            r=subprocess.run(["python", os.path.join(BASE,"train_models.py")],
                capture_output=True,text=True,cwd=BASE)
        if r.returncode==0:
            st.success("✅ Both models trained and saved!")
            with st.expander("📋 Training output & metrics"):
                st.code(r.stdout[-4000:] if len(r.stdout)>4000 else r.stdout or "(no output)")
            st.cache_resource.clear()
            st.info("🔄 Page cache cleared — refresh the page to load the new models.")
            st.rerun()
        else:
            st.error("❌ train_models.py failed:")
            st.code(r.stderr[-2000:])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Settings":
    st.markdown('<div class="section-title">⚙️ Settings</div>', unsafe_allow_html=True)
    t1,t2=st.tabs(["🌱 Seed Demo Data","📱 SMS Config"])

    with t1:
        st.markdown("Populate the database with demo patients and AI-generated predictions for testing.")
        n=st.slider("Number of demo patients",5,30,10)
        if st.button("🌱 Seed Demo Patients",use_container_width=True):
            if not model_ready:
                st.error("Models must be loaded first.")
            else:
                import random
                FIRST=["Amina","Wanjiku","Aisha","Grace","Mary","Fatuma","Joyce","Esther","Lilian","Zawadi"]
                LAST=["Mwangi","Odhiambo","Kamau","Kimani","Otieno","Njenga","Wanjiru","Achieng","Njoroge","Hassan"]
                bar=st.progress(0,text="Seeding…")
                for i in range(n):
                    pid=str(uuid.uuid4())[:8].upper()
                    nm=f"{random.choice(FIRST)} {random.choice(LAST)}"
                    ph=f"+2547{''.join([str(random.randint(0,9)) for _ in range(8)])}"
                    ag=random.randint(18,42); wk=random.randint(8,38)
                    db.upsert_patient(pid,nm,ph,ag,wk,random.randint(0,4),bool(random.randint(0,1)))
                    for _ in range(random.randint(1,3)):
                        sb=random.randint(100,175); db2=random.randint(60,110)
                        bs=round(random.uniform(4.5,14.0),1); hb=round(random.uniform(7.5,14.0),1)
                        vid=db.save_visit(pid,sb,db2,bs,round(random.uniform(97.5,100.5),1),
                                           random.randint(60,110),round(random.uniform(50,95),1))
                        res=predict(age=ag,systolic_bp=sb,diastolic_bp=db2,blood_sugar=bs,
                                    body_temp=98.2,heart_rate=76,hemoglobin_gdl=hb,
                                    gravidity=random.randint(1,4),gestational_age_weeks=float(wk))
                        db.save_prediction(vid,pid,res["risk_level"],res["risk_score"],
                                            res["top_risks"],res["shap_reasons"],res["suggested_action"])
                    dy=3 if res["risk_level"]=="high risk" else 14 if res["risk_level"]=="mid risk" else 28
                    db.schedule_appointment(pid,(date.today()+timedelta(days=dy)).isoformat(),f"Follow-up: {res['risk_level']}")
                    bar.progress((i+1)/n,text=f"Seeded {i+1}/{n}…")
                st.success(f"✅ {n} demo patients seeded!"); st.rerun()

    with t2:
        st.markdown("""
        Create a `.env` file in the app folder to enable real SMS:
        ```env
        AT_USERNAME=your_username
        AT_API_KEY=your_api_key
        CLINIC_PHONE=+254700000000
        NURSE_PHONES=+254700000001,+254700000002
        ```
        Without credentials the app runs in **mock mode** — SMS are logged to the database and printed to the terminal.
        Get a free sandbox key at [account.africastalking.com](https://account.africastalking.com)
        """)
        st.info("🔵 Current mode: MOCK — SMS logged to DB only.")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.covariance import EllipticEnvelope
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(page_title="DiabetesAI Pro", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);min-height:100vh;}
.glass{background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.15);border-radius:16px;padding:1.5rem;backdrop-filter:blur(12px);margin-bottom:1rem;}
.mrow{display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1rem;}
.mc{flex:1;min-width:110px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.18);border-radius:12px;padding:.9rem;text-align:center;}
.mv{font-size:1.6rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.ml{font-size:.68rem;color:rgba(255,255,255,0.5);letter-spacing:.06em;text-transform:uppercase;margin-top:3px;}
[data-testid="stSidebar"]{background:rgba(15,12,41,0.9)!important;border-right:1px solid rgba(255,255,255,0.1);}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
h1{background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.2rem!important;font-weight:800!important;}
h2,h3{color:#c4b5fd!important;}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:rgba(255,255,255,0.05);border-radius:10px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#94a3b8;font-weight:600;}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,#7c3aed,#2563eb)!important;color:white!important;}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#2563eb);color:white;border:none;border-radius:10px;padding:.6rem 2rem;font-weight:700;font-size:1rem;transition:all .3s;width:100%;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(124,58,237,.5);}
p,label,.stMarkdown{color:#cbd5e1!important;}
.sum-box{border-radius:14px;padding:1.2rem 1.5rem;margin:.5rem 0;border-left:4px solid;}
.sum-low{background:rgba(52,211,153,0.12);border-color:#34d399;}
.sum-med{background:rgba(251,191,36,0.12);border-color:#fbbf24;}
.sum-high{background:rgba(248,113,113,0.12);border-color:#f87171;}
.sum-crit{background:rgba(239,68,68,0.18);border-color:#ef4444;}
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING & PREPROCESSING ─────────────
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

@st.cache_data
def preprocess(df):
    d = df.copy()
    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for c in zero_cols:
        d[c] = d[c].replace(0, np.nan)
        d[c] = d.groupby("Outcome")[c].transform(lambda x: x.fillna(x.median()))
    # Outlier removal
    try:
        ee = EllipticEnvelope(contamination=0.05, random_state=42)
        mask = ee.fit_predict(d.drop("Outcome", axis=1)) == 1
        d = d[mask].reset_index(drop=True)
    except:
        pass
    # Feature engineering
    d["Glucose_BMI"]         = d["Glucose"] * d["BMI"] / 100
    d["Age_Preg"]            = d["Age"] * d["Pregnancies"]
    d["Insulin_Glucose"]     = d["Insulin"] / (d["Glucose"] + 1)
    d["BMI_Age"]             = d["BMI"] * d["Age"] / 100
    d["Glucose_squared"]     = d["Glucose"] ** 2 / 10000
    d["Risk_score"]          = (d["Glucose"]/200 + d["BMI"]/67 + d["Age"]/81) / 3
    return d

@st.cache_resource
def train_model(df):
    feat_cols = [c for c in df.columns if c != "Outcome"]
    X, y = df[feat_cols], df["Outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # SMOTE on training data
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_tr_s, y_tr)

    # Base models (well-tuned)
    rf  = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=3,
                                  min_samples_leaf=1, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    et  = ExtraTreesClassifier(n_estimators=400, max_depth=10, random_state=42,
                                class_weight="balanced", n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                      max_depth=5, subsample=0.8, random_state=42)
    xgb = XGBClassifier(n_estimators=400, learning_rate=0.04, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                         reg_lambda=1.5, eval_metric="logloss",
                         random_state=42, verbosity=0)
    svm = SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski")

    # Stacking with meta-learner
    stacker = StackingClassifier(
        estimators=[("rf",rf),("et",et),("gb",gb),("xgb",xgb),("svm",svm),("knn",knn)],
        final_estimator=LogisticRegression(C=2.0, max_iter=2000, random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=True, n_jobs=-1
    )
    stacker.fit(X_res, y_res)

    y_pred  = stacker.predict(X_te_s)
    y_proba = stacker.predict_proba(X_te_s)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_te, y_pred)*100, 2),
        "precision": round(precision_score(y_te, y_pred)*100, 2),
        "recall":    round(recall_score(y_te, y_pred)*100, 2),
        "f1":        round(f1_score(y_te, y_pred)*100, 2),
        "roc_auc":   round(roc_auc_score(y_te, y_proba)*100, 2),
    }

    # CV score
    cv_scores = cross_val_score(stacker, scaler.transform(X), y,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                 scoring="accuracy", n_jobs=-1)
    metrics["cv_mean"] = round(cv_scores.mean()*100, 2)
    metrics["cv_std"]  = round(cv_scores.std()*100, 2)

    rf.fit(X_res, y_res)
    importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)
    cm = confusion_matrix(y_te, y_pred)
    return stacker, scaler, metrics, importances, cm, feat_cols

# ── LOAD & TRAIN ─────────────────────────────
raw   = load_data()
clean = preprocess(raw)
with st.spinner("🔬 Training advanced stacking ensemble (SMOTE + 6 models)…"):
    model, scaler, metrics, importances, cm, feat_cols = train_model(clean)

# ── HEADER ───────────────────────────────────
st.markdown("<h1>🩺 DiabetesAI Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;margin-top:-12px;'>Stacking Ensemble · SMOTE · 6 Base Models · Outlier Removal · Feature Engineering</p>", unsafe_allow_html=True)

st.markdown(f"""
<div class="mrow">
  <div class="mc"><div class="mv">{metrics['accuracy']}%</div><div class="ml">Test Accuracy</div></div>
  <div class="mc"><div class="mv">{metrics['precision']}%</div><div class="ml">Precision</div></div>
  <div class="mc"><div class="mv">{metrics['recall']}%</div><div class="ml">Recall</div></div>
  <div class="mc"><div class="mv">{metrics['f1']}%</div><div class="ml">F1 Score</div></div>
  <div class="mc"><div class="mv">{metrics['roc_auc']}%</div><div class="ml">ROC-AUC</div></div>
  <div class="mc"><div class="mv">{metrics['cv_mean']}%</div><div class="ml">CV Score</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict & Summary", "📊 Data Explorer", "📈 Model Performance", "ℹ️ About"])


# ═══════════════════════════════════════════
# HEALTH SUMMARY GENERATOR
# ═══════════════════════════════════════════
def get_health_summary(prob, glucose, bmi, age, blood_pressure, insulin, dpf, pregnancies):
    pct = prob * 100
    findings, recommendations = [], []

    if glucose > 140:
        findings.append(f"🔴 **Glucose critically high** ({glucose} mg/dL — normal: 70–99 fasting)")
        recommendations.append("Consult an endocrinologist immediately for blood sugar management.")
    elif glucose > 100:
        findings.append(f"🟡 **Glucose slightly elevated** ({glucose} mg/dL — borderline)")
        recommendations.append("Reduce refined carbohydrates and sugary beverages.")
    else:
        findings.append(f"🟢 **Glucose normal** ({glucose} mg/dL)")

    if bmi >= 30:
        findings.append(f"🔴 **Obese BMI** ({bmi} — normal: 18.5–24.9)")
        recommendations.append("Aim for 5–7% weight loss through diet and 150 min/week exercise.")
    elif bmi >= 25:
        findings.append(f"🟡 **Overweight BMI** ({bmi})")
        recommendations.append("Focus on portion control and regular moderate exercise.")
    else:
        findings.append(f"🟢 **Healthy BMI** ({bmi})")

    if blood_pressure > 90:
        findings.append(f"🔴 **High blood pressure** ({blood_pressure} mm Hg)")
        recommendations.append("Reduce sodium intake and monitor BP regularly.")
    elif blood_pressure > 80:
        findings.append(f"🟡 **Borderline blood pressure** ({blood_pressure} mm Hg)")
    else:
        findings.append(f"🟢 **Blood pressure normal** ({blood_pressure} mm Hg)")

    if insulin > 200:
        findings.append(f"🔴 **Insulin very high** ({insulin} μU/mL) — possible insulin resistance")
        recommendations.append("Get an insulin resistance panel test done.")
    elif insulin < 16:
        findings.append(f"🟡 **Insulin low** ({insulin} μU/mL)")
    else:
        findings.append(f"🟢 **Insulin within range** ({insulin} μU/mL)")

    if dpf > 1.0:
        findings.append(f"🔴 **Strong family history** of diabetes (DPF: {dpf})")
        recommendations.append("Given strong hereditary risk, get HbA1c checked every 6 months.")
    elif dpf > 0.5:
        findings.append(f"🟡 **Moderate family history** (DPF: {dpf})")
    else:
        findings.append(f"🟢 **Low hereditary risk** (DPF: {dpf})")

    if age >= 45:
        findings.append(f"⚠️ **Age risk factor** ({age} yrs — risk rises significantly after 45)")
        recommendations.append("Annual fasting glucose tests recommended for your age group.")

    # Overall verdict
    if pct < 25:
        verdict_class = "sum-low"
        verdict = f"✅ LOW RISK ({pct:.1f}%) — Your indicators look healthy."
        overall = "Your overall health profile is good. Maintain a balanced diet and active lifestyle to keep diabetes risk minimal."
    elif pct < 50:
        verdict_class = "sum-med"
        verdict = f"⚠️ MODERATE RISK ({pct:.1f}%) — Some indicators need attention."
        overall = "You have some risk factors. Adopting lifestyle changes now can significantly reduce your chances of developing diabetes."
    elif pct < 75:
        verdict_class = "sum-high"
        verdict = f"🔶 HIGH RISK ({pct:.1f}%) — Multiple concerning indicators."
        overall = "Your risk profile is high. Please consult a doctor and get an HbA1c and fasting glucose test done soon."
    else:
        verdict_class = "sum-crit"
        verdict = f"🚨 CRITICAL RISK ({pct:.1f}%) — Immediate medical attention advised."
        overall = "Multiple severe risk factors detected. Please seek medical evaluation as soon as possible. Early diagnosis can prevent serious complications."

    if not recommendations:
        recommendations.append("Continue your healthy lifestyle — you're doing great!")

    return verdict_class, verdict, overall, findings, recommendations


# ═══════════════════════════════════════════
# TAB 1 — PREDICT & SUMMARY
# ═══════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🧬 Patient Parameters")
        pregnancies    = st.slider("Pregnancies", 0, 20, 1)
        glucose        = st.slider("Glucose (mg/dL)", 44, 200, 120)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 24, 122, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 7, 99, 20)
        insulin        = st.slider("Insulin (μU/mL)", 14, 846, 80)
        bmi            = st.slider("BMI", 18.0, 67.1, 32.0, step=0.1)
        dpf            = st.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.47, step=0.01)
        age            = st.slider("Age", 21, 81, 33)
        predict_btn    = st.button("🔍 Predict & Generate Health Summary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass" style="min-height:420px;">', unsafe_allow_html=True)
        st.subheader("📋 Risk Assessment")

        if predict_btn:
            inp = pd.DataFrame([{
                "Pregnancies": pregnancies, "Glucose": glucose,
                "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
                "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age,
                "Glucose_BMI": glucose * bmi / 100,
                "Age_Preg": age * pregnancies,
                "Insulin_Glucose": insulin / (glucose + 1),
                "BMI_Age": bmi * age / 100,
                "Glucose_squared": glucose ** 2 / 10000,
                "Risk_score": (glucose/200 + bmi/67 + age/81) / 3,
            }])[feat_cols]

            prob = model.predict_proba(scaler.transform(inp))[0][1]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob*100, 1),
                number={"suffix":"%","font":{"color":"#a78bfa","size":34}},
                title={"text":"Diabetes Risk Score","font":{"color":"#e2e8f0","size":13}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#94a3b8"},
                    "bar":{"color":"#f87171" if prob>0.5 else "#34d399"},
                    "steps":[
                        {"range":[0,25],"color":"rgba(52,211,153,0.15)"},
                        {"range":[25,50],"color":"rgba(251,191,36,0.12)"},
                        {"range":[50,75],"color":"rgba(248,113,113,0.12)"},
                        {"range":[75,100],"color":"rgba(239,68,68,0.18)"},
                    ],
                    "threshold":{"line":{"color":"white","width":3},"value":50},
                    "bgcolor":"rgba(0,0,0,0)"
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0",height=240,margin=dict(t=40,b=0,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""<div style='text-align:center;padding:3rem 1rem;'>
              <div style='font-size:4rem;'>🔮</div>
              <p style='color:#64748b;'>Set parameters and click <strong style="color:#a78bfa">Predict</strong></p>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── HEALTH SUMMARY SECTION ──
    if predict_btn:
        st.markdown("---")
        st.subheader("📝 Personalised Health Summary")

        prob_val = model.predict_proba(scaler.transform(inp))[0][1]
        v_class, verdict, overall, findings, recs = get_health_summary(
            prob_val, glucose, bmi, age, blood_pressure, insulin, dpf, pregnancies
        )

        st.markdown(f'<div class="sum-box {v_class}"><h3 style="margin:0">{verdict}</h3><p style="margin:.5rem 0 0;">{overall}</p></div>', unsafe_allow_html=True)

        col_f, col_r2 = st.columns(2)
        with col_f:
            st.markdown("#### 🔬 Key Findings")
            for f in findings:
                st.markdown(f"- {f}")
        with col_r2:
            st.markdown("#### 💊 Recommendations")
            for i, r in enumerate(recs, 1):
                st.markdown(f"**{i}.** {r}")

        # Radar chart
        st.markdown("#### 📡 Risk Factor Radar")
        cats  = ["Glucose","BMI","Blood Pressure","Insulin","Age","Family History"]
        vals  = [
            min(glucose/200, 1), min(bmi/67, 1),
            min(blood_pressure/122, 1), min(insulin/846, 1),
            min(age/81, 1), min(dpf/2.42, 1)
        ]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor="rgba(167,139,250,0.2)",
            line=dict(color="#a78bfa", width=2),
            name="Patient"
        ))
        fig_r.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True, range=[0,1], color="#94a3b8"),
                       angularaxis=dict(color="#e2e8f0")),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            height=340, margin=dict(t=20,b=20,l=20,r=20)
        )
        st.plotly_chart(fig_r, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ═══════════════════════════════════════════
with tab2:
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Records", len(raw))
    c2.metric("Diabetic", int(raw["Outcome"].sum()))
    c3.metric("Non-Diabetic", int((raw["Outcome"]==0).sum()))

    if st.checkbox("Show raw dataset"):
        st.dataframe(raw, use_container_width=True)

    feat = st.selectbox("Feature distribution", [c for c in raw.columns if c != "Outcome"])
    fig_h = px.histogram(raw, x=feat, color="Outcome", barmode="overlay", nbins=30,
                         color_discrete_map={0:"#34d399",1:"#f87171"}, template="plotly_dark")
    fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("🔥 Correlation Heatmap")
    corr = clean.corr()
    fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", template="plotly_dark")
    fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_c, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════
with tab3:
    st.subheader("🏆 Model Metrics")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, lbl, v in zip([c1,c2,c3,c4,c5,c6],
                            ["Accuracy","Precision","Recall","F1","ROC-AUC","CV Mean"],
                            [metrics["accuracy"],metrics["precision"],metrics["recall"],
                             metrics["f1"],metrics["roc_auc"],metrics["cv_mean"]]):
        col.metric(lbl, f"{v}%")

    ca, cb = st.columns(2)
    with ca:
        st.subheader("🎯 Feature Importance")
        fig_i = go.Figure(go.Bar(x=importances.values, y=importances.index,
                                  orientation="h",
                                  marker=dict(color=importances.values, colorscale="Plasma")))
        fig_i.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0",height=380)
        st.plotly_chart(fig_i, use_container_width=True)

    with cb:
        st.subheader("🗂️ Confusion Matrix")
        fig_cm = px.imshow(cm, labels=dict(x="Predicted",y="Actual"),
                            x=["Not Diabetic","Diabetic"],y=["Not Diabetic","Diabetic"],
                            text_auto=True, color_continuous_scale="Purples", template="plotly_dark")
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""
    ### 🤖 Model Architecture
    | Layer | Model | Role |
    |---|---|---|
    | Base | 🌲 Random Forest (500 trees) | Bagging ensemble |
    | Base | 🌳 Extra Trees (400 trees) | High variance capture |
    | Base | 📈 Gradient Boosting (300 est.) | Sequential correction |
    | Base | ⚡ XGBoost (400 est., regularized) | Best single model |
    | Base | 🔮 SVM (RBF, C=3) | Margin maximisation |
    | Base | 📍 KNN (k=7, distance weight) | Local pattern matching |
    | Meta | 🧠 Logistic Regression | Optimal combination |
    
    **Techniques:** SMOTE oversampling · RobustScaler · Outlier removal (EllipticEnvelope) · 6 engineered features · 5-fold stratified CV
    """)


# ═══════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════
with tab4:
    st.markdown("""
    ## 🩺 About DiabetesAI Pro
    
    ### 🔬 Data Preprocessing
    - **Zero-value imputation** using class-stratified medians (Glucose, BMI, Insulin, etc.)
    - **Outlier removal** using EllipticEnvelope (Mahalanobis distance-based)
    - **RobustScaler** — resistant to outliers (uses IQR instead of mean/std)
    
    ### 🧪 Feature Engineering  
    6 new features derived from domain knowledge: Glucose×BMI, Age×Pregnancies, Insulin/Glucose ratio, BMI×Age, Glucose², and a composite Risk Score.
    
    ### 🤖 SMOTE + Stacking Ensemble
    - **SMOTE** balances the training set by synthesising new diabetic patient profiles
    - **Stacking** uses 6 diverse base models, with a Logistic Regression meta-learner to optimally combine their predictions
    
    ### 📊 Dataset
    Pima Indians Diabetes Database — 768 records, 8 clinical features.
    
    ### ⚠️ Disclaimer  
    For educational purposes only. Not a substitute for professional medical diagnosis.
    """)

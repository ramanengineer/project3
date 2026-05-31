"""
NZ Road Crash Severity Predictor
297.201 Project 3 — Streamlit Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="NZ Crash Predictor", page_icon="🚗", layout="wide")

@st.cache_resource
def load_everything():
    model   = joblib.load('best_model.pkl')
    encoder = joblib.load('label_encoder.pkl')
    with open('feature_columns.json') as f:
        cols = json.load(f)
    with open('feature_meta.json') as f:
        meta = json.load(f)
    return model, encoder, cols, meta

try:
    model, encoder, feature_cols, meta = load_everything()
except Exception as e:
    st.error(f"Could not load model files.\n\n{e}")
    st.stop()

st.title("🚗 NZ Road Crash Severity Predictor")
st.markdown("""
**297.201 Data Science — Project 3** | Data: Waka Kotahi CAS Open Data

Predicts whether a crash will be **Minor**, **Serious**, or **Fatal**
based on road and weather conditions. Adjust the sliders on the left to try different scenarios.
""")

st.sidebar.header("Set Crash Conditions")

num_cols  = meta.get('num_cols', [])
cat_cols  = meta.get('cat_cols', [])
cat_opts  = meta.get('cat_options', {})
num_range = meta.get('num_ranges', {})

user_inputs = {}

for col in num_cols:
    r = num_range.get(col, {'min': 0, 'max': 100, 'mean': 50})
    user_inputs[col] = st.sidebar.slider(
        col.replace('_', ' ').title(),
        min_value=float(r['min']),
        max_value=float(r['max']),
        value=float(r['mean']),
        step=1.0
    )

for col in cat_cols:
    options = cat_opts.get(col, ['Unknown'])
    user_inputs[col] = st.sidebar.selectbox(col.replace('_', ' ').title(), options)

def build_input(user_inputs, feature_cols, cat_cols):
    row = {col: 0 for col in feature_cols}
    for col, val in user_inputs.items():
        if col in row:
            row[col] = val
    for col in cat_cols:
        val = user_inputs.get(col, 'Unknown')
        ohe_col = f"{col}_{val}"
        if ohe_col in row:
            row[ohe_col] = 1
    return pd.DataFrame([row])[feature_cols]

input_df   = build_input(user_inputs, feature_cols, cat_cols)
proba      = model.predict_proba(input_df)[0]
pred_idx   = int(np.argmax(proba))
pred_label = encoder.classes_[pred_idx]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    icons = {"Fatal": "🔴", "Serious": "🟠", "Minor": "🟢"}
    st.metric("Predicted Severity", f"{icons.get(pred_label,'⚪')} {pred_label}")

    st.subheader("Confidence")
    prob_df = pd.DataFrame({
        'Severity': encoder.classes_,
        'Probability': proba
    }).sort_values('Probability', ascending=False)

    fig, ax = plt.subplots(figsize=(4, 3))
    colors = ['#d62728' if c == 'Fatal' else '#ff7f0e' if c == 'Serious' else '#2ca02c'
              for c in prob_df['Severity']]
    bars = ax.barh(prob_df['Severity'], prob_df['Probability'], color=colors)
    ax.bar_label(bars, labels=[f'{p*100:.0f}%' for p in prob_df['Probability']], padding=3)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("What factors matter most?")
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('clf', None), 'feature_importances_'):
        feat_imp = pd.Series(model.named_steps['clf'].feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
    else:
        feat_imp = None

    if feat_imp is not None:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        feat_imp.sort_values().plot(kind='barh', ax=ax2, color='steelblue')
        ax2.set_title('Top 15 Features Driving Crash Severity')
        ax2.set_xlabel('Importance Score')
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()
    else:
        st.info("Feature importance is available when Random Forest or Gradient Boosting is the best model.")

st.markdown("---")
st.subheader("Key Insights")
c1, c2, c3 = st.columns(3)
with c1:
    st.info("🚀 **Speed is the #1 factor**\n\nHigher speed limits strongly predict fatal crashes.")
with c2:
    st.warning("🌙 **Night crashes are deadlier**\n\nFewer crashes at night but far more likely to be fatal.")
with c3:
    st.success("🌲 **Road type matters**\n\nCurves and hills lead to more serious outcomes.")

st.markdown("---")
st.caption("Data: Waka Kotahi NZ Transport Agency — CAS Open Data | CC-BY 4.0")
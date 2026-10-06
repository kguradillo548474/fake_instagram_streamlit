import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

st.set_page_config(page_title="Fake Instagram Profile Detector", page_icon="🔎", layout="wide")

BASE = Path(__file__).resolve().parent
FEATURES = [
    "profile pic", "nums/length username", "fullname words",
    "nums/length fullname", "name==username", "description length",
    "external URL", "private", "#posts", "#followers", "#follows",
    "followers_following_ratio"
]
TARGET = "fake"

MODEL_PATH = BASE / "research_model.joblib"
META_PATH = BASE / "model_metadata.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def load_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))

model = load_model()
meta = load_metadata()

st.title("Fake Instagram Profile Detector")
st.caption("Verified XGBoost research prototype — Group 17")

with st.expander("Important: what this prototype can and cannot do", expanded=True):
    st.info(
        "This is a profile-risk screening prototype, not an official Instagram verification tool. "
        "It does not log into Instagram, bypass access controls, scrape private data, or ban/report accounts. "
        "It predicts from the 12 structured features used by the research dataset. The output is a model score, "
        "not a calibrated real-world probability or proof that an account is fake."
    )

with st.sidebar:
    st.header("Decision setting")
    threshold = st.slider(
        "Fake classification threshold", 0.10, 0.90, 0.50, 0.01,
        help="Lower thresholds can increase recall but may increase false positives."
    )
    st.markdown("---")
    st.header("Loaded model configuration")
    bp = meta["model_params"]
    st.write(f"n_estimators: {bp['n_estimators']}")
    st.write(f"max_depth: {bp['max_depth']}")
    st.write(f"learning_rate: {bp['learning_rate']}")
    st.write(f"subsample: {bp['subsample']}")
    st.write(f"colsample_bytree: {bp['colsample_bytree']}")
    st.write("random_state: 42")

def model_label(score):
    return "Likely Fake" if score >= threshold else "Likely Genuine"

def contributions(row):
    try:
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(row)
        values = booster.predict(dmatrix, pred_contribs=True)[0]
        return pd.DataFrame({
            "feature": FEATURES,
            "contribution": values[:-1],
        }).sort_values("contribution", key=np.abs, ascending=False)
    except Exception:
        return None

tab1, tab2, tab3 = st.tabs(["Single Profile", "Batch Screening", "Research Verification"])

with tab1:
    st.subheader("Manual profile screening")
    c1, c2, c3 = st.columns(3)
    with c1:
        profile_pic = st.selectbox(
            "Profile picture present?",
            ["Yes", "No"],
            help="Select Yes if the account has a profile photo; No if it is blank or uses a generic/default image."
        )
        username_digit_ratio = st.number_input(
            "Username digit ratio (digits ÷ total username length, e.g. alex123 = 3/8 = 0.375)",
            0.0, 1.0, 0.0, 0.01,
            help="Example: 'alex123' has 3 digits out of 8 characters, so enter 0.375."
        )
        fullname_words = st.number_input(
            "Full-name word count (count words, e.g. 'John Smith' = 2)",
            0, 100, 2, 1,
            help="Count how many separate words are in the account name."
        )
        fullname_digit_ratio = st.number_input(
            "Full-name digit ratio (digits ÷ total full-name length, e.g. John2 = 1/6 ≈ 0.17)",
            0.0, 1.0, 0.0, 0.01,
            help="Example: 'John2' has 1 digit in a 6-character name, so the ratio is about 0.17."
        )
    with c2:
        same_name = st.selectbox(
            "Full name equals username?",
            ["Yes", "No"],
            help="Choose Yes only if the account name matches the username exactly (for example: username = 'johnsmith', name = 'johnsmith')."
        )
        bio_length = st.number_input(
            "Bio/description length (number of characters in the bio)",
            0, 10000, 0, 1,
            help="Count the number of characters in the bio text. Empty bio = 0."
        )
        external_url = st.selectbox(
            "External URL present?",
            ["Yes", "No"],
            help="Choose Yes if the profile includes a website URL in the bio."
        )
        private = st.selectbox(
            "Account is private?",
            ["Yes", "No"],
            help="Choose Yes if the account is private; No if it is public."
        )
    with c3:
        posts = st.number_input(
            "Post count (total number of posts)",
            0, 1000000, 0, 1,
            help="Enter the total number of posts on the account."
        )
        followers = st.number_input(
            "Follower count (total followers)",
            0, 1000000000, 0, 1,
            help="Enter how many followers the account has."
        )
        follows = st.number_input(
            "Following count (accounts they follow)",
            0, 1000000000, 0, 1,
            help="Enter how many accounts they follow."
        )
        ratio_default = followers / max(follows, 1)
        ratio = st.number_input(
            "Follower/following ratio (followers ÷ following, e.g. 5000 ÷ 200 = 25)",
            0.0, 1000000000.0,
            float(min(ratio_default, 1e9)), 0.01,
            help="This is a ratio, not a percentage. For example, 5000 followers and 200 following = 25."
        )

    if st.button("Analyze profile", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "profile pic": int(profile_pic == "Yes"),
            "nums/length username": username_digit_ratio,
            "fullname words": fullname_words,
            "nums/length fullname": fullname_digit_ratio,
            "name==username": int(same_name == "Yes"),
            "description length": bio_length,
            "external URL": int(external_url == "Yes"),
            "private": int(private == "Yes"),
            "#posts": posts,
            "#followers": followers,
            "#follows": follows,
            "followers_following_ratio": ratio,
        }], columns=FEATURES)
        score = float(model.predict_proba(row)[0, 1])
        if score >= threshold:
            st.error(f"### {model_label(score)}")
        else:
            st.success(f"### {model_label(score)}")
        a, b = st.columns(2)
        a.metric("Fake-class score", f"{score:.2%}")
        b.metric("Decision threshold", f"{threshold:.0%}")
        st.caption("Use this as a screening signal. It is not a verified probability.")
        c = contributions(row)
        if c is not None:
            st.subheader("Model-level signal breakdown")
            st.dataframe(c, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Batch screening")
    st.write("Upload a CSV containing the 12 research features. The `fake` column is optional.")
    batch = st.file_uploader("Upload profile CSV", type=["csv"], key="batch")
    if batch is not None:
        batch_df = pd.read_csv(batch)
        missing = [c for c in FEATURES if c not in batch_df.columns]
        if missing:
            st.error("Missing required columns:")
            st.code("\n".join(missing))
        else:
            bx = batch_df[FEATURES].apply(pd.to_numeric, errors="coerce")
            valid = bx.notna().all(axis=1)
            result = batch_df.loc[valid].copy()
            bx = bx.loc[valid]
            scores = model.predict_proba(bx)[:, 1]
            result["fake_score"] = scores
            result["model_decision"] = np.where(scores >= threshold, "Likely Fake", "Likely Genuine")
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download screening results",
                result.to_csv(index=False).encode("utf-8"),
                "instagram_screening_results.csv",
                "text/csv"
            )
            if TARGET in result.columns:
                yt = pd.to_numeric(result[TARGET], errors="coerce")
                valid_y = yt.isin([0, 1])
                if valid_y.sum() > 1 and yt[valid_y].nunique() == 2:
                    pred = (scores[valid_y.to_numpy()] >= threshold).astype(int)
                    truth = yt[valid_y].astype(int)
                    st.subheader("Batch evaluation")
                    cols = st.columns(5)
                    cols[0].metric("Accuracy", f"{accuracy_score(truth, pred):.2%}")
                    cols[1].metric("Precision", f"{precision_score(truth, pred, zero_division=0):.2%}")
                    cols[2].metric("Recall", f"{recall_score(truth, pred, zero_division=0):.2%}")
                    cols[3].metric("F1", f"{f1_score(truth, pred, zero_division=0):.2%}")
                    cols[4].metric("ROC-AUC", f"{roc_auc_score(truth, scores[valid_y.to_numpy()]):.4f}")

with tab3:
    st.subheader("Research verification")
    st.write("The app uses the fitted 500-tree model artifact. This model was trained using one fixed configuration, not grid search.")
    st.write(f"Clean dataset rows: **{meta['clean_rows']:,}**")
    st.write(f"Training rows: **{meta['train_rows']:,}**")
    st.write(f"Test rows: **{meta['test_rows']:,}**")
    st.write(f"Training split: **{meta['training_split']}**")
    st.write(f"Random state: **{bp['random_state']}**")

    st.markdown("#### Held-out test result")
    m = meta["test_metrics"]
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{m['accuracy']:.2%}")
    cols[1].metric("Precision", f"{m['precision']:.2%}")
    cols[2].metric("Recall", f"{m['recall']:.2%}")
    cols[3].metric("F1", f"{m['f1']:.2%}")
    cols[4].metric("ROC-AUC", f"{m['roc_auc']:.4f}")

    st.markdown("#### Previously reported tuned XGBoost model confusion matrix")
    st.caption("This is the earlier tuned-model result, not the result for the currently loaded 500-tree model.")
    st.dataframe(
        pd.DataFrame(
            meta["previous_tuned_model_confusion_matrix"],
            index=["Genuine", "Fake"],
            columns=["Pred Genuine", "Pred Fake"],
        )
    )

    st.markdown("#### Feature order")
    st.code("\n".join(FEATURES))
    st.caption(f"XGBoost version used to create the bundled artifact: {xgb.__version__}")

st.markdown("---")
st.caption("Academic prototype • Model output is a risk signal • Human review is recommended for consequential decisions")

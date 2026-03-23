# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SocioBuzz - app.py
# Run: streamlit run app.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import base64
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# sklearn version compatibility patch
# The .joblib artifacts were saved with sklearn 1.6.1 which
# introduced _RemainderColsList. This shim makes older sklearn
# versions able to load those files without error.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    from sklearn.compose._column_transformer import _RemainderColsList  # noqa: F401
except ImportError:
    import sklearn.compose._column_transformer as _ct

    class _RemainderColsList(list):
        """Compatibility shim for sklearn < 1.2 loading artifacts from >= 1.2."""
        def __init__(self, columns=None, future_dtype=None):
            super().__init__(columns or [])
            self.future_dtype = future_dtype

    _ct._RemainderColsList = _RemainderColsList

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PAGE CONFIG
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
st.set_page_config(
    page_title = "SocioBuzz",
    page_icon = "",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLOBAL CSS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #0f1117;
    border-right: 1px solid #2e2e3a;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 6px 0;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
div[data-testid="metric-container"] {
    background-color: #1e1e2e;
    border: 1px solid #2e2e3a;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: #9e9eb0 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 22px !important;
    font-weight: 700;
}
.section-card {
    background-color: #1e1e2e;
    border: 1px solid #2e2e3a;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.badge-high {
    background: #1a472a; color: #69db7c; border: 1px solid #2f9e44;
    border-radius: 8px; padding: 6px 16px; font-weight: 700;
    font-size: 18px; display: inline-block;
}
.badge-medium {
    background: #3d2c00; color: #ffd43b; border: 1px solid #f59f00;
    border-radius: 8px; padding: 6px 16px; font-weight: 700;
    font-size: 18px; display: inline-block;
}
.badge-low {
    background: #3b0a0a; color: #ff6b6b; border: 1px solid #c92a2a;
    border-radius: 8px; padding: 6px 16px; font-weight: 700;
    font-size: 18px; display: inline-block;
}
.badge-viral {
    background: #0d3b6e; color: #74c0fc; border: 1px solid #1971c2;
    border-radius: 8px; padding: 6px 16px; font-weight: 700;
    font-size: 18px; display: inline-block;
}
.badge-notviral {
    background: #2e2e2e; color: #adb5bd; border: 1px solid #495057;
    border-radius: 8px; padding: 6px 16px; font-weight: 700;
    font-size: 18px; display: inline-block;
}
.section-heading {
    font-size: 13px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #7c7c99;
    margin-bottom: 12px; margin-top: 4px;
}
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #6c63ff, #48cae4);
    color: white !important; font-size: 16px; font-weight: 700;
    border-radius: 10px; border: none; padding: 14px;
    width: 100%; transition: opacity 0.2s;
}
div[data-testid="stFormSubmitButton"] button:hover { opacity: 0.88; }
hr { border: none; border-top: 1px solid #2e2e3a; margin: 20px 0; }
button[data-baseweb="tab"] { font-size: 14px; font-weight: 600; color: #9e9eb0; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #6c63ff !important;
    border-bottom: 2px solid #6c63ff !important;
}
</style>
""", unsafe_allow_html=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Paths & VADER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base_path = os.path.dirname(os.path.abspath(__file__))
Other_imp_byproducts = os.path.join(Base_path, "Other_imp_byproducts")
sia = SentimentIntensityAnalyzer()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Artifacts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(os.path.join(Other_imp_byproducts, "preprocessor.joblib"))
    model_lr = joblib.load(os.path.join(Other_imp_byproducts, "model.joblib"))
    model_xgb = joblib.load(os.path.join(Other_imp_byproducts, "xgb_model.joblib"))
    rf_reg = joblib.load(os.path.join(Other_imp_byproducts, "rf_regressor.joblib"))
    le = joblib.load(os.path.join(Other_imp_byproducts, "label_encoder.joblib"))
    meta = joblib.load(os.path.join(Other_imp_byproducts, "inference_meta.joblib"))
    preprocessor_ex = joblib.load(os.path.join(Other_imp_byproducts, "preprocessor_ex.joblib"))
    extreme_xgb = joblib.load(os.path.join(Other_imp_byproducts, "extreme_xgb_tuned.joblib"))
    meta_ex = joblib.load(os.path.join(Other_imp_byproducts, "inference_meta_ex.joblib"))
    return (preprocessor, model_lr, model_xgb, rf_reg, le, meta,
            preprocessor_ex, extreme_xgb, meta_ex)

(preprocessor, model_lr, model_xgb, rf_reg, le, meta,
 preprocessor_ex, extreme_xgb, meta_ex) = load_artifacts()

CAT_COLS = meta["cat_cols_ml"]
NUM_COLS = meta["num_cols_ml"]
LABEL_CLASSES = meta["label_classes"]
KNOWN_HASHTAGS = meta["known_hashtags"]
RAW_MEDIANS = meta["raw_medians"]
RAW_MODES = meta["raw_modes"]
LOW_THRESH = meta["low_thresh"]
HIGH_THRESH = meta["high_thresh"]

Q25 = meta_ex["q25"]
Q75 = meta_ex["q75"]
CAT_COLS_EX = meta_ex["cat_cols_ex"]
NUM_COLS_EX = meta_ex["num_cols_ex"]
RAW_MEDIANS_EX = meta_ex.get("raw_medians_ex", RAW_MEDIANS)
RAW_MODES_EX = meta_ex.get("raw_modes_ex",   RAW_MODES)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Known Hashtags
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_known_hashtags(prep, cat_cols):
    try:
        ohe = prep.named_transformers_["cat"]
        ht_idx = list(cat_cols).index("primary_hashtag")
        return set(ohe.categories_[ht_idx])
    except Exception:
        return set()

KNOWN_HT_MAIN = get_known_hashtags(preprocessor,    CAT_COLS)
KNOWN_HT_EX = get_known_hashtags(preprocessor_ex, CAT_COLS_EX)
KNOWN_HT_ALL = KNOWN_HT_MAIN | KNOWN_HT_EX
FALLBACK_HASHTAG = "#fyp"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@st.cache_data
def load_sample():
    path = os.path.join(Other_imp_byproducts, "engagement_sample.csv")
    df   = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

@st.cache_data
def load_viral_rates():
    path = os.path.join(Other_imp_byproducts, "platform_hashtag_viral_rates.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_feature_importance():
    path = os.path.join(Other_imp_byproducts, "feature_importance_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_merged_size():
    path = os.path.join(Base_path, "Datasets", "merged_dataset.csv")
    if os.path.exists(path):
        return len(pd.read_csv(path))
    return 16418  # fallback

sample_df = load_sample()
viral_rates_df = load_viral_rates()
fi_df = load_feature_importance()
merged_size = load_merged_size()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_primary_hashtag(s):
    if not s or not str(s).strip(): return "#Other"
    for p in [p.strip() for p in str(s).split(",")]:
        p = p if p.startswith("#") else f"#{p}"
        if p != "#": return p
    return "#Other"

def count_hashtags(s):
    if not s or not str(s).strip(): return 1
    return max(1, str(s).count(",") + 1)

def compute_sentiment(text):
    if not text or not str(text).strip():
        return 0.0, 0.5, "Neutral", "Unknown"
    scores = sia.polarity_scores(str(text))
    vc = scores["compound"]
    sentiment_score = (vc + 1) / 2
    if vc > 0.5:     return vc, sentiment_score, "Positive", "Joy"
    elif vc > 0.05:  return vc, sentiment_score, "Positive", "Surprise"
    elif vc < -0.5:  return vc, sentiment_score, "Negative", "Anger"
    elif vc < -0.05: return vc, sentiment_score, "Negative", "Sadness"
    else:            return vc, sentiment_score, "Neutral",  "Unknown"

def infer_topic(text):
    if not text or not str(text).strip(): return "General"
    t = text.lower()
    if any(w in t for w in ["price","cost","discount","offer","deal","sale","promo"]):
        return "Pricing"
    elif any(w in t for w in ["product","launch","new","feature","release","buy","review",
                               "unbox","gaming","tech","apple","samsung","asus"]):
        return "Product"
    elif any(w in t for w in ["help","support","issue","problem","complaint","fix",
                               "broken","error","refund","return"]):
        return "CustomerService"
    elif any(w in t for w in ["event","live","join","webinar","concert","show","attend"]):
        return "Event"
    elif any(w in t for w in ["announce","exciting","breaking","news","update",
                               "introducing","reveal","just dropped","just launched"]):
        return "Announcement"
    return "General"

def resolve_hashtag_from_raw(hashtags_raw, known_ht_set):
    if not hashtags_raw or not str(hashtags_raw).strip():
        return "#Other", False, None
    all_tags = []
    for p in [p.strip() for p in str(hashtags_raw).split(",")]:
        tag = p if p.startswith("#") else f"#{p}"
        if tag != "#": all_tags.append(tag)
    for tag in all_tags:
        if tag in known_ht_set:
            return tag, (tag != all_tags[0]), all_tags[0]
    return FALLBACK_HASHTAG, True, all_tags[0] if all_tags else "#Other"

def build_row(inputs):
    row = {}
    for c in NUM_COLS: row[c] = float(inputs.get(c, RAW_MEDIANS.get(c, 0.0)))
    for c in CAT_COLS: row[c] = str(inputs.get(c, RAW_MODES.get(c, "Unknown")))
    return pd.DataFrame([row])

def predict_engagement(inputs):
    # Main 3-class prediction
    row = build_row(inputs)
    X_in = preprocessor.transform(row)
    pred_lr = le.inverse_transform(model_lr.predict(X_in))[0]
    pred_xgb = le.inverse_transform(model_xgb.predict(X_in))[0]
    prob_xgb = model_xgb.predict_proba(X_in)[0]
    pred_reg = rf_reg.predict(X_in)[0]

    # Extreme binary prediction
    row_ex_dict = {}
    for c in CAT_COLS_EX:
        row_ex_dict[c] = str(inputs.get(c, RAW_MODES_EX.get(c, "Unknown")))
    for c in NUM_COLS_EX:
        row_ex_dict[c] = float(inputs.get(c, RAW_MEDIANS_EX.get(c, 0.0)))

    row_ex = pd.DataFrame([row_ex_dict])
    row_ex = row_ex[CAT_COLS_EX + NUM_COLS_EX]

    X_ex_in = preprocessor_ex.transform(row_ex)
    pred_viral = extreme_xgb.predict(X_ex_in)[0]
    prob_viral_xgb = extreme_xgb.predict_proba(X_ex_in)[0][1]

    return pred_lr, pred_xgb, prob_xgb, pred_reg, pred_viral, prob_viral_xgb

def get_best_posting_time(platform, sample_df):
    if sample_df.empty or "platform" not in sample_df.columns:
        return None, None, None, None
    plat_df = sample_df[sample_df["platform"] == platform].copy()
    if plat_df.empty: plat_df = sample_df.copy()
    if "timestamp" not in plat_df.columns: return None, None, None, None
    plat_df["hour"] = plat_df["timestamp"].dt.hour
    plat_df["weekday"] = plat_df["timestamp"].dt.day_name()
    hour_avg = plat_df.groupby("hour")["engagement_rate"].mean()
    day_avg = plat_df.groupby("weekday")["engagement_rate"].mean()
    best_hour = int(hour_avg.idxmax())
    best_day = day_avg.idxmax()
    return best_hour, best_day, hour_avg.max(), day_avg.max()

def format_hour(h):
    if h == 0: 
        return "12:00 AM"
    elif h < 12:
        return f"{h}:00 AM"
    elif h == 12:
        return "12:00 PM"
    else:
        return f"{h-12}:00 PM"

def level_badge(level):
    cls = {"High":"badge-high","Medium":"badge-medium","Low":"badge-low"}.get(level,"badge-low")
    return f'<span class="{cls}">{level}</span>'

def viral_badge(prob):
    if prob >= 0.60:
        return '<span class="badge-viral">Likely Viral</span>'
    return '<span class="badge-notviral">Not Viral</span>'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SIDEBAR
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
with st.sidebar:
    _logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Diagrams", "SocioBuzz_Logo.png")
    if os.path.exists(_logo_path):
        with open(_logo_path, "rb") as _f:
            _logo_b64 = base64.b64encode(_f.read()).decode()
        st.markdown(f"""
        <div style='text-align:center; padding: 10px 0 16px 0;'>
            <img src='data:image/png;base64,{_logo_b64}'
                 style='width:120px; height:120px; border-radius:50%;
                        object-fit:cover; border:2px solid #6c63ff;'/>
            <div style='font-size:12px; color:#6c6c88; margin-top:8px;'>
                Social Media Trend Predictor
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navigation",
        ["Home", "Predict", "Analytics", "About"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("""
    <div style='font-size:11px; color:#4e4e66; text-align:center; padding-top:8px;'>
        University of London FYP 2026<br>
        <span style='color:#6c63ff;'>SocioBuzz v1.0</span>
    </div>
    """, unsafe_allow_html=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HOME PAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if "Home" in page:

    st.markdown("""
    <div style='margin-bottom:8px;'>
        <span style='font-size:36px; font-weight:800;'>SocioBuzz</span>
        <span style='font-size:16px; color:#6c6c88; margin-left:12px;'>
            Social Media Trend Prediction Dashboard
        </span>
    </div>
    <div style='color:#9e9eb0; font-size:15px; margin-bottom:28px;'>
        Predict engagement level and viral likelihood of a post
        <b style='color:#6c63ff;'>before</b> it goes live.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Platforms", "6")
    c2.metric("Dataset Size", f"~{merged_size}")
    c3.metric("Features", str(len(CAT_COLS) + len(NUM_COLS)))
    c4.metric("Models", "4")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-card'>
        <div class='section-heading'>How it works</div>
        <ol style='color:#c0c0d0; font-size:14px; line-height:2;'>
            <li>Type your post content and hashtags</li>
            <li>Select platform, content type and campaign details</li>
            <li>Sentiment, emotion and topic are <b style='color:#6c63ff;'>auto-detected</b> from your text</li>
            <li>Model predicts Engagement Level: <b>Low / Medium / High</b></li>
            <li>Extreme Binary classifier predicts <b>Viral vs Not Viral</b> (top 25%)</li>
            <li>Random Forest Regressor estimates the <b>engagement rate</b></li>
            <li>Personalised <b>posting tips</b> are generated from platform data</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Platform Engagement Overview")

    if not sample_df.empty:
        avg_plat = (sample_df.groupby("platform")["engagement_rate"]
                    .mean().reset_index()
                    .rename(columns={"engagement_rate":"avg_engagement_rate"})
                    .sort_values("avg_engagement_rate", ascending=True))
        fig = px.bar(
            avg_plat, x="platform", y="avg_engagement_rate",
            color="platform",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"avg_engagement_rate":"Avg Engagement Rate","platform":"Platform"},
        )
        fig.update_layout(
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="#c0c0d0", showlegend=False,
            xaxis=dict(gridcolor="#2e2e3a"),
            yaxis=dict(gridcolor="#2e2e3a"),
            bargap=0.3,
            bargroupgap=0.1,
        )
        fig.update_xaxes(tickmode="array",
                 tickvals=avg_plat["platform"].tolist(),
                 ticktext=avg_plat["platform"].tolist())
        st.plotly_chart(fig, use_container_width=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PREDICT PAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
elif "Predict" in page:

    st.markdown("""
    <div style='margin-bottom:20px;'>
        <span style='font-size:30px; font-weight:800;'>Predict Post Engagement</span><br>
        <span style='color:#9e9eb0; font-size:14px;'>
            Fill in your post details below. Sentiment, emotion and topic are computed automatically.
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):

        st.markdown('<div class="section-heading">Post Content</div>',
                    unsafe_allow_html=True)
        post_text = st.text_area(
            "Post Text", value="",
            placeholder = "Write your post here — sentiment, emotion and topic are auto-detected from this.",
            height=110, label_visibility="collapsed"
        )
        hashtags_raw = st.text_input(
            "Hashtags", value="",
            placeholder="Hashtags (comma separated)  e.g.  #fyp, #trending, #viral",
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Post Settings</div>',
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            platform = st.selectbox("Platform",
                                   ["Twitter","Instagram","Facebook","YouTube","Reddit","TikTok"])
            content_type = st.selectbox("Content Type",
                                       ["Post","Video","Story","Reel","Thread","Clip"])
            topic_category = st.selectbox("Topic Category",
                                         ["Auto-detect","General","Pricing","Product",
                                          "CustomerService","Event","Announcement"])
        with col2:
            day_of_week = st.selectbox("Day of Week",
                                      ["Monday","Tuesday","Wednesday","Thursday",
                                       "Friday","Saturday","Sunday"])
            hour_of_day = st.slider("Hour of Day (0–23)", 0, 23, 12)
            region = st.selectbox("Region",
                                 ["US","UK","Global","IN","AU","CA","EU","Unknown"])
        with col3:
            brand_name = st.text_input("Brand Name",    value="Unknown")
            campaign_name = st.text_input("Campaign Name", value="Unknown")
            campaign_phase = st.selectbox("Campaign Phase",
                                         ["Launch","Mid","End","Unknown"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "**Note:** Features `user_engagement_growth` and `buzz_change_rate` require "
            "historical account-level data and are not available at prediction time. "
            "They are set to **0.0** (training median) for inference — a known limitation "
            "acknowledged in the project evaluation."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    # Results
    if submitted:
        if not post_text or not str(post_text).strip():
            st.warning("No post text entered — prediction is based on context signals only (platform, timing, region). Add post text for a more accurate result.")
        vader_compound, sentiment_score, sentiment_label, emotion_type = compute_sentiment(post_text)
        inferred_topic = infer_topic(post_text)
        final_topic = inferred_topic if topic_category == "Auto-detect" else topic_category
        toxicity_score = 0.05

        primary_hashtag_raw = extract_primary_hashtag(hashtags_raw)
        num_of_hashtags = count_hashtags(hashtags_raw)
        hashtag_len = len(hashtags_raw) if hashtags_raw else 0

        hashtag_for_model, is_oov, original_hashtag = resolve_hashtag_from_raw(
            hashtags_raw, KNOWN_HT_ALL)

        inputs = {
            "platform":                platform,
            "day_of_week":             day_of_week,
            "hour_of_day":             hour_of_day,
            "content_type":            content_type,
            "region":                  region,
            "topic_category":          final_topic,
            "brand_name":              brand_name,
            "campaign_name":           campaign_name,
            "campaign_phase":          campaign_phase,
            "primary_hashtag":         hashtag_for_model,
            "num_of_hashtags":         num_of_hashtags,
            "hashtag_len":             hashtag_len,
            "sentiment_score":         sentiment_score,
            "sentiment_label":         sentiment_label,
            "emotion_type":            emotion_type,
            "toxicity_score":          toxicity_score,
            "vader_compound":          vader_compound,
            "user_past_sentiment_avg": 0.0,
            "user_engagement_growth":  0.0,
            "buzz_change_rate":        0.0,
        }

        (pred_lr, pred_xgb, prob_xgb,
         pred_reg, pred_viral, prob_viral_xgb) = predict_engagement(inputs)

        # OOV Warning
        if is_oov:
            st.warning(
                f"**`{original_hashtag}`** is not in the model vocabulary. "
                f"Substituted **`{hashtag_for_model}`** for prediction. "
                f"Use hashtags from **Analytics -> Viral Hashtags** for best results."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Auto-detected Signals
        st.markdown("<div class='section-heading'>Auto-detected Signals</div>",
                    unsafe_allow_html=True)
        s1,s2,s3,s4,s5,s6,s7 = st.columns(7)
        s1.metric("Sentiment", sentiment_label)
        s2.metric("VADER Score", f"{vader_compound:.3f}")
        s3.metric("Emotion", emotion_type)
        s4.metric("Topic", final_topic)
        s5.metric("Primary Hashtag", primary_hashtag_raw)
        s6.metric("Hashtag Count", num_of_hashtags)
        s7.metric("Hashtag Length", hashtag_len)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Results
        st.markdown("<div class='section-heading'>Prediction Results</div>",
                    unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.markdown(f"""
            <div class='section-card' style='text-align:center;'>
                <div style='font-size:11px;color:#7c7c99;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:10px;'>
                    Logistic Regression
                </div>
                {level_badge(pred_lr)}
            </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class='section-card' style='text-align:center;'>
                <div style='font-size:11px;color:#7c7c99;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:10px;'>
                    XGBoost
                </div>
                {level_badge(pred_xgb)}
            </div>""", unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class='section-card' style='text-align:center;'>
                <div style='font-size:11px;color:#7c7c99;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:10px;'>
                    Predicted Eng. Rate
                </div>
                <span style='font-size:26px;font-weight:800;color:#fff;'>
                    {pred_reg:.4f}
                </span>
            </div>""", unsafe_allow_html=True)

        with r4:
            st.markdown(f"""
            <div class='section-card' style='text-align:center;'>
                <div style='font-size:11px;color:#7c7c99;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:10px;'>
                    Viral Status
                </div>
                {viral_badge(prob_viral_xgb)}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Viral gauge
        st.markdown("<div class='section-heading'>Viral Probability</div>",
                    unsafe_allow_html=True)

        gauge_color = (
            "#27ae60" if prob_viral_xgb >= 0.75
            else "#f39c12" if prob_viral_xgb >= 0.60
            else "#e74c3c"
        )
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = round(prob_viral_xgb * 100, 1),
            delta = {"reference": 60, "valueformat": ".1f",
                     "increasing": {"color":"#27ae60"},
                     "decreasing": {"color":"#e74c3c"}},
            title = {"text": "Viral Probability (%)",
                     "font": {"color":"#c0c0d0", "size":16}},
            number = {"suffix": "%", "font": {"color":"#ffffff","size":42}},
            gauge = {
                "axis":  {"range":[0,100], "tickcolor":"#4e4e66",
                          "tickfont":{"color":"#9e9eb0"}},
                "bar":   {"color": gauge_color, "thickness":0.25},
                "bgcolor":"#1e1e2e",
                "bordercolor":"#2e2e3a",
                "steps": [
                    {"range":[0,  60], "color":"#2a1a1a"},
                    {"range":[60, 75], "color":"#2a2210"},
                    {"range":[75,100], "color":"#0d2a1a"},
                ],
                "threshold": {
                    "line":{"color":"#6c63ff","width":3},
                    "thickness":0.75, "value":60
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#1e1e2e", font_color="#c0c0d0",
            height=260, margin=dict(t=60, b=20, l=40, r=40)
        )
        g1, g2, g3 = st.columns([1,2,1])
        with g2:
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style='text-align:center; color:#7c7c99; font-size:13px; margin-top:-10px;'>
            Viral threshold: top 25% of engagement rate &gt;
            <b style='color:#fff;'>{Q75:.4f}</b>
            &nbsp;|&nbsp; Viral probability:
            <b style='color:#fff;'>{prob_viral_xgb*100:.1f}%</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Posting tips
        st.markdown("<div class='section-heading'>Posting Recommendations</div>",
                    unsafe_allow_html=True)

        best_hour, best_day, best_hour_rate, _ = get_best_posting_time(platform, sample_df)

        if best_hour is not None:
            if hour_of_day != best_hour:
                st.warning(f"**Posting Time** — Consider posting at **{format_hour(best_hour)}** "
                           f"for best results on {platform}. "
                           f"Posts at that hour average {best_hour_rate:.3f} engagement rate.")
            else:
                st.success(f"**Posting Time** — Great timing! "
                           f"**{format_hour(hour_of_day)}** is near peak engagement for {platform}.")

        if vader_compound < -0.05:
            st.warning("**Sentiment** — Negative tone detected. "
                       "Posts with positive sentiment tend to get higher engagement.")
        elif vader_compound > 0.5:
            st.success("**Sentiment** — Strong positive tone. This tends to perform well.")
        else:
            st.info("**Sentiment** — Neutral tone. Adding enthusiasm may improve engagement.")

        if prob_viral_xgb >= 0.75:
            st.success(f"**Viral Potential** — High ({prob_viral_xgb*100:.1f}%). "
                       f"Strong signals — publish soon.")
        elif prob_viral_xgb >= 0.60:
            st.info(f"**Viral Potential** — Moderate ({prob_viral_xgb*100:.1f}%). "
                    f"Refine hashtags or sentiment to push higher.")
        else:
            st.warning(f"**Viral Potential** — Low ({prob_viral_xgb*100:.1f}%). "
                       f"Try adjusting hashtags, timing or tone.")

        if pred_reg >= HIGH_THRESH:
            st.success(f"**Engagement Rate** — Predicted {pred_reg:.4f} is "
                       f"above the high threshold ({HIGH_THRESH:.4f}). Strong post.")
        elif pred_reg >= LOW_THRESH:
            st.info(f"**Engagement Rate** — Predicted {pred_reg:.4f} is in the "
                    f"medium range. Refining hashtags or timing could push this higher.")
        else:
            st.warning(f"**Engagement Rate** — Predicted {pred_reg:.4f} is below "
                       f"the low threshold ({LOW_THRESH:.4f}). Consider revising content.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Best hour chart
        if not sample_df.empty and "timestamp" in sample_df.columns and best_hour is not None:
            st.markdown(f"<div class='section-heading'>Best Posting Window — {platform}</div>",
                        unsafe_allow_html=True)
            bh1, bh2 = st.columns(2)
            bh1.metric("Best Hour", format_hour(best_hour))
            bh2.metric("Best Day",  best_day)

            plat_df = sample_df[sample_df["platform"] == platform].copy()
            if plat_df.empty: plat_df = sample_df.copy()
            plat_df["hour"] = plat_df["timestamp"].dt.hour
            hour_avg = plat_df.groupby("hour")["engagement_rate"].mean().reset_index()

            fig3 = px.line(hour_avg, x="hour", y="engagement_rate", markers=True,
                           labels={"hour":"Hour of Day",
                                   "engagement_rate":"Avg Engagement Rate"})
            fig3.add_vline(x=hour_of_day, line_dash="dash", line_color="#f39c12",
                           annotation_text="Your hour", annotation_position="top right")
            fig3.add_vline(x=best_hour, line_dash="dash", line_color="#27ae60",
                           annotation_text="Best hour", annotation_position="top left")
            fig3.update_layout(
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="#c0c0d0",
                xaxis=dict(gridcolor="#2e2e3a", dtick=1),
                yaxis=dict(gridcolor="#2e2e3a"),
            )
            st.plotly_chart(fig3, use_container_width=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ANALYTICS PAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
elif "Analytics" in page:

    st.markdown("<span style='font-size:30px;font-weight:800;'>Analytics & Insights</span>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Platform Trends",
        "#️Viral Hashtags",
        "Engagement Timeline",
        "Feature Importance"
    ])

    PLOTLY_LAYOUT = dict(
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#c0c0d0",
        xaxis=dict(gridcolor="#2e2e3a"),
        yaxis=dict(gridcolor="#2e2e3a"),
    )

    with tab1:
        st.subheader("Average Engagement Rate by Platform")
        if not sample_df.empty:
            avg_plat = (sample_df.groupby("platform")["engagement_rate"]
                        .agg(["mean","std","count"]).reset_index())
            avg_plat.columns = ["Platform","Mean","Std","Count"]
            fig = px.bar(avg_plat, x="Platform", y="Mean", error_y="Std",
                         color="Platform",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(avg_plat.sort_values("Mean", ascending=False),
                         use_container_width=True)

    with tab2:
        st.subheader("Top Hashtags by Viral Rate (Model-Compatible)")
        if not viral_rates_df.empty:
            top_n = st.slider("Show top N hashtags", 5, 30, 15)
            vr_filtered = viral_rates_df[
                viral_rates_df["primary_hashtag"].isin(KNOWN_HT_ALL)
            ].copy()
            if vr_filtered.empty:
                st.warning("No viral hashtag data matches model vocabulary.")
                vr_filtered = viral_rates_df.copy()
            else:
                st.caption(f"Showing {len(vr_filtered)} model-compatible hashtags "
                           f"(of {len(viral_rates_df)} total).")
            top_ht = vr_filtered.sort_values("viral_rate", ascending=False).head(top_n)
            fig = px.bar(top_ht, x="primary_hashtag", y="viral_rate",
                         color="platform",
                         color_discrete_sequence=px.colors.qualitative.Vivid,
                         labels={"primary_hashtag":"Hashtag","viral_rate":"Viral Rate"})
            fig.update_xaxes(tickangle=45)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_ht, use_container_width=True)
        else:
            st.info("Viral rates data not available.")

    with tab3:
        st.subheader("Monthly Engagement Rate by Platform")
        if not sample_df.empty and "timestamp" in sample_df.columns:
            df_time = sample_df.dropna(subset=["timestamp"]).copy()
            df_time["month"] = df_time["timestamp"].dt.to_period("M").astype(str)
            monthly = (df_time.groupby(["month","platform"])["engagement_rate"]
                       .mean().reset_index())
            fig = px.line(monthly, x="month", y="engagement_rate",
                          color="platform", markers=True,
                          color_discrete_sequence=px.colors.qualitative.Vivid,
                          labels={"engagement_rate":"Avg Engagement Rate","month":"Month"})
            fig.update_xaxes(tickangle=45)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Top 20 Feature Importances — XGBoost Tuned")
        if not fi_df.empty:
            fig = px.bar(fi_df.sort_values("importance"),
                         x="importance", y="feature", orientation="h",
                         color="importance",
                         color_continuous_scale="Viridis",
                         labels={"importance":"Importance Score","feature":"Feature"})
            fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not available.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ABOUT PAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
elif "About" in page:

    st.markdown("<span style='font-size:30px;font-weight:800;'>About SocioBuzz</span>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-card'>
        <div class='section-heading'>Project</div>
        <p style='color:#c0c0d0; font-size:14px; line-height:1.8;'>
            SocioBuzz is a Final Year Project system that predicts whether a social media post
            will achieve high, medium or low engagement <b>before it is published</b>,
            using only signals available at the time of posting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Models")
    st.markdown("""
| Model | Task |
|---|---|
| Logistic Regression | Multi-class engagement level (Low / Medium / High) |
| XGBoost (tuned, GridSearchCV) | Multi-class engagement level |
| Random Forest Regressor (tuned) | Continuous engagement rate prediction |
| XGBoost Tuned (Extreme, 5-Fold CV) | Viral vs Not Viral (top / bottom 25%) |
    """)

    st.markdown("### What the user provides")
    st.markdown("""
| Input | How it is used |
|---|---|
| Post text | Auto-computes VADER score, sentiment label, emotion type, topic |
| Hashtags | Auto-computes primary hashtag, count, length |
| Platform, Content Type, Region | Passed directly to model |
| Day of Week, Hour of Day | Temporal features |
| Topic Category | Auto-detected or manually overridden |
| Brand, Campaign Name, Phase | Campaign context features |
    """)

    st.markdown("### What is auto-computed")
    st.markdown("""
| Signal | Source |
|---|---|
| `vader_compound` | VADER sentiment analysis |
| `sentiment_score` | Scaled VADER: (compound + 1) / 2 |
| `sentiment_label` | Derived from VADER score |
| `emotion_type` | Derived from VADER score |
| `topic_category` | Keyword inference from post text |
| `num_of_hashtags` | Counted from hashtag input |
| `hashtag_len` | Length of hashtag string |
| `primary_hashtag` | First hashtag extracted |
| `toxicity_score` | Fixed at 0.05 (safe default) |
| `user_engagement_growth` | Fixed at 0.0 — requires historical account data (not available at inference) |
| `buzz_change_rate` | Fixed at 0.0 — requires historical trend data (not available at inference) |
    """)

    st.markdown("""
    <div style='text-align:center; margin-top:40px; color:#4e4e66; font-size:13px;'>
        SocioBuzz — University of London FYP 2026 &nbsp;|&nbsp;
        <span style='color:#6c63ff;'>v1.0</span>
    </div>
    """, unsafe_allow_html=True)
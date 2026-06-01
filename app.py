import streamlit as st
import pickle
import requests
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Spam Detection Hub",
    page_icon="📩",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
}

.big-title{
    text-align:center;
    font-size:55px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#cbd5e1;
}

.result-card{
    padding:20px;
    border-radius:20px;
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(10px);
    border:1px solid rgba(255,255,255,0.15);
}

.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_resources():

    lstm_model = load_model(
        "lstm_model.keras"
    )

    bilstm_model = load_model(
        "bilstm_model.keras"
    )

    with open(
        "tokenizer.pkl",
        "rb"
    ) as f:

        tokenizer = pickle.load(f)

    return (
        lstm_model,
        bilstm_model,
        tokenizer
    )

lstm_model, bilstm_model, tokenizer = load_resources()

MAX_LEN = 50

# =====================================
# PREDICTION FUNCTION
# =====================================

def predict(model, text):

    seq = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prob = model.predict(
        padded,
        verbose=0
    )[0][0]

    label = (
        "🚨 SPAM"
        if prob > 0.5
        else "✅ HAM"
    )

    confidence = (
        prob
        if prob > 0.5
        else 1 - prob
    )

    return label, confidence

# =====================================
# HEADER
# =====================================

st.markdown(
    '<p class="big-title">📩 AI Spam Detection Hub</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Comparing LSTM vs Bidirectional LSTM</p>',
    unsafe_allow_html=True
)

st.write("")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📌 Project Info")

st.sidebar.success("""
SMS Spam Detection

Models Used:

✔ LSTM

✔ Bidirectional LSTM

Dataset:

SMS Spam Collection Dataset
""")

st.sidebar.markdown("---")

st.sidebar.write(
    "Enter an SMS message and compare both models."
)

# =====================================
# INPUT
# =====================================

message = st.text_area(
    "✍️ Enter SMS Message",
    height=180,
    placeholder="Example: Congratulations! You won a free iPhone..."
)

model_choice = st.radio(
    "Choose Prediction Mode",
    [
        "LSTM",
        "BiLSTM",
        "Compare Both"
    ]
)

# =====================================
# BUTTON
# =====================================

if st.button("🚀 Detect Message"):

    if message.strip() == "":

        st.warning(
            "Please enter a message."
        )

    else:

        # -----------------------
        # LSTM ONLY
        # -----------------------

        if model_choice == "LSTM":

            label, confidence = predict(
                lstm_model,
                message
            )

            st.success(
                f"Prediction: {label}"
            )

            st.progress(
                float(confidence)
            )

            st.write(
                f"Confidence: {confidence:.2%}"
            )

        # -----------------------
        # BILSTM ONLY
        # -----------------------

        elif model_choice == "BiLSTM":

            label, confidence = predict(
                bilstm_model,
                message
            )

            st.success(
                f"Prediction: {label}"
            )

            st.progress(
                float(confidence)
            )

            st.write(
                f"Confidence: {confidence:.2%}"
            )

        # -----------------------
        # COMPARE BOTH
        # -----------------------

        else:

            lstm_label, lstm_conf = predict(
                lstm_model,
                message
            )

            bilstm_label, bilstm_conf = predict(
                bilstm_model,
                message
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "## 🔹 LSTM"
                )

                st.metric(
                    "Prediction",
                    lstm_label
                )

                st.progress(
                    float(lstm_conf)
                )

                st.write(
                    f"Confidence: {lstm_conf:.2%}"
                )

            with col2:

                st.markdown(
                    "## 🔹 BiLSTM"
                )

                st.metric(
                    "Prediction",
                    bilstm_label
                )

                st.progress(
                    float(bilstm_conf)
                )

                st.write(
                    f"Confidence: {bilstm_conf:.2%}"
                )

            winner = (
                "BiLSTM"
                if bilstm_conf > lstm_conf
                else "LSTM"
            )

            st.success(
                f"🏆 Higher Confidence Model: {winner}"
            )

            # -----------------------
            # GAUGE CHART
            # -----------------------

            spam_score = max(
                lstm_conf,
                bilstm_conf
            ) * 100

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=spam_score,
                    title={
                        'text': "Confidence Score"
                    },
                    gauge={
                        'axis': {
                            'range': [0,100]
                        }
                    }
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown(
    """
    <div class='footer'>
    Built with Streamlit • LSTM vs BiLSTM Spam Detection
    </div>
    """,
    unsafe_allow_html=True
)
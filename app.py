import streamlit as st
import pandas as pd
import plotly.express as px
import time
from src.predict import predict_purchase

# PAGE CONFIG

st.set_page_config(page_title="AI Purchase Prediction", page_icon="🛒", layout="wide")


# MODERN CSS (FIXED INPUT STYLE)

st.markdown("""
<style>
/* Background aplikasi */
.stApp { background: linear-gradient(135deg, #e0f7fa 0%, #e8f5e9 100%); color: #004d40; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #006064 !important; color: white !important; }
[data-testid="stSidebar"] * { color: white !important; }

/* Judul */
.title { font-size: 32px; font-weight: 800; color: #004d40 !important; }

/* Desain Kotak Input  */
div[data-baseweb="input"] {
    background-color: #ffffff !important; 
    border: 1px solid #006064 !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"] input {
    color: #004d40 !important;
    font-weight: 700 !important;
    -webkit-text-fill-color: #004d40 !important;
}
/* Memastikan label terlihat jelas */
label {
    font-weight: 600 !important;
    color: #004d40 !important;
}

/* Box Kontainer */
.main-box {
    background-color: rgba(255, 255, 255, 0.8);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #b2dfdb;
    margin-bottom: 20px;
}

/* Tombol */
.stButton button {
    background: linear-gradient(90deg, #00acc1, #43a047);
    color: white !important;
    font-weight: 800;
    border-radius: 12px;
    height: 50px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)



# SIDEBAR

st.sidebar.title("🧠 About System")

st.sidebar.info(
    """
    Dashboard ini menggunakan Machine Learning
    untuk memprediksi apakah pelanggan akan
    melakukan pembelian atau tidak.
    """
)

st.sidebar.markdown("---")

st.sidebar.header("🎯 Model Performance")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.metric(
        label="Accuracy",
        value="87.35%"
    )

with col2:
    st.metric(
        label="F1 Score",
        value="64.55%"
    )

col3, col4 = st.sidebar.columns(2)

with col3:
    st.metric(
        label="Precision",
        value="57.03%"
    )

with col4:
    st.metric(
        label="Recall",
        value="74.35%"
    )

st.sidebar.info(
    """
🤖 **Algorithm**

Random Forest Classifier

🌲 Ensemble Learning Model
"""
)


# MAIN CONTENT

st.markdown('<p class="title">🛒 AI Customer Purchase Prediction</p>', unsafe_allow_html=True)
st.write("Isi formulir di bawah ini untuk melihat hasil prediksi.")

with st.container():
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    
    # Grid Input
    col1, col2, col3 = st.columns(3)
    with col1:
        administrative = st.number_input("Admin Pages", 0, value=2)
        administrative_duration = st.number_input("Admin Duration", 0.0, value=50.0)
    with col2:
        informational = st.number_input("Info Pages", 0, value=1)
        informational_duration = st.number_input("Info Duration", 0.0, value=20.0)
    with col3:
        product_related = st.number_input("Product Pages", 0, value=40)
        product_related_duration = st.number_input("Product Duration", 0.0, value=800.0)

col4, col5, col6 = st.columns(3)
with col4:
    bounce_rates = st.number_input(
        "Bounce Rate",
        0.0,
        1.0,
        0.01
    )

    exit_rates = st.number_input(
        "Exit Rate",
        0.0,
        1.0,
        0.03
    )

with col5:
    page_values = st.number_input(
        "Page Value",
        min_value=0.0,
        max_value=50.0,
        value=20.0
    )

    special_day_option = st.selectbox(
        "Promotion Period",
        [
            "Normal Day",
            "Special Event"
        ]
    )

    special_day = (
        1.0
        if special_day_option == "Special Event"
        else 0.0
    )

with col6:
    month = st.selectbox(
        "Month",
        [
            "Feb",
            "Mar",
            "May",
            "June",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec"
        ]
    )

    visitor_type = st.selectbox(
        "Visitor Type",
        [
            "Returning_Visitor",
            "New_Visitor",
            "Other"
        ]
    )

run_btn = st.button("🚀 Run Prediction")

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# HASIL PREDIKSI

if run_btn:
    with st.spinner("Analyzing..."):
        time.sleep(1)
        prediction, probability = predict_purchase(
            administrative, administrative_duration, informational, informational_duration,
            product_related, product_related_duration, bounce_rates, exit_rates,
            page_values, special_day, month, 2, 2, 1, 3, visitor_type, True
        )

    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.subheader("📊 Prediksi Hasil")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Purchase Prob.", f"{probability*100:.1f}%")
    m2.metric("Confidence", "High" if probability > 0.6 else "Moderate")
    m3.metric("Verdict", "BUY" if prediction == 1 else "NO BUY")
    
    fig = px.pie(values=[probability*100, (1-probability)*100], names=["Purchase", "Not Purchase"],
                 hole=0.6, color_discrete_sequence=["#43a047", "#00acc1"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#004d40", showlegend=True)
    st.plotly_chart(fig, use_container_width=False)
    
    if prediction == 1:
        st.success("✅ Pelanggan kemungkinan besar akan melakukan pembelian.")
    else:
        st.error("❌ Pelanggan kemungkinan besar tidak melakukan pembelian.")
    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #EAF2FA, #F4F7FC, #E3EDF8);
    background-attachment: fixed;
}

.block-container {
    max-width: 950px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(120deg, #102B50, #1D5B91);
    padding: 30px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 5px 18px rgba(16, 43, 80, 0.15);
}

.hero h1 {
    color: white;
    font-size: 30px;
}

.hero p {
    color: #DCEBFA;
    font-size: 15px;
}

.gov-banner {
    background: rgba(255, 255, 255, 0.92);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    margin: 20px 0;
    border: 1px solid #DFE7F0;
    box-shadow: 0 4px 14px rgba(30, 70, 110, 0.06);
}

.gov-banner h3 {
    color: #173B65;
}

.gov-banner p {
    color: #526579;
}

.feature {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #E0E7EF;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(30, 70, 110, 0.06);
}

.feature h3 {
    color: #1D5B91;
}

.footer {
    text-align: center;
    color: #718096;
    padding: 20px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Government Schemes & Policies Chatbot</h1>
    <p>Your assistant for government schemes, policies, benefits and eligibility.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="gov-banner">
    <h3>🇮🇳 Welcome to Government Information Assistant</h3>
    <p>Explore information about public welfare schemes,
    government policies and citizen benefits.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature">
        <h3>📋</h3>
        <b>Government Schemes</b>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature">
        <h3>⚖️</h3>
        <b>Government Policies</b>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature">
        <h3>🎯</h3>
        <b>Citizen Benefits</b>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.caption("Developed by DARSHAN")

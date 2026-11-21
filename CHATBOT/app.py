import streamlit as st

from RAG_engine import load_base, ask_rag


# Page Configuration
st.set_page_config(
    page_title="Government Schemes & Policies Chatbot",
    page_icon="🇮🇳",
    layout="centered"
)


# Frontend Design
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #EAF2FA,
        #F4F7FC,
        #E3EDF8
    );
    background-attachment: fixed;
}

.block-container {
    max-width: 950px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(
        120deg,
        #102B50,
        #1D5B91
    );
    padding: 30px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
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

.stChatMessage {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="hero">
    <h1>Government Schemes & Policies Chatbot</h1>
    <p>
        Your assistant for government schemes,
        policies, benefits and eligibility.
    </p>
</div>
""", unsafe_allow_html=True)


# Welcome
st.markdown("""
<div class="gov-banner">
    <h3>🇮🇳 Welcome to Government Information Assistant</h3>
    <p>
        Explore information about public welfare schemes,
        government policies and citizen benefits.
    </p>
</div>
""", unsafe_allow_html=True)


# Features
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


# API Key
try:

    mykey = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it in Streamlit Cloud Secrets."
    )

    st.stop()


# Load RAG
@st.cache_resource
def initiate_rag():

    return load_base()


try:

    with st.spinner("Reading government PDF documents..."):

        index, embed_model, chunks = initiate_rag()

    st.success(
        f"Government PDF documents loaded successfully! "
        f"{len(chunks)} chunks created."
    )

except Exception as e:

    st.error(f"Unable to load PDF documents: {e}")

    st.stop()


# Chat History
if "chat_hist" not in st.session_state:

    st.session_state.chat_hist = []


# Clear Chat
if st.button("Clear Chat"):

    st.session_state.chat_hist = []

    st.rerun()


# Display Previous Messages
for qa in st.session_state.chat_hist:

    with st.chat_message(qa["role"]):

        st.markdown(qa["content"])


# Question Input
question = st.chat_input(
    "Ask about government schemes and policies..."
)


# Generate Answer
if question:

    st.session_state.chat_hist.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Finding information from government documents..."):

            try:

                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=mykey
                )

                st.markdown(answer)

                st.session_state.chat_hist.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"Unable to generate an answer: {e}"
                )


# Footer
st.markdown("""
<div class="footer">
    Government Schemes & Policies Assistant | Powered by RAG
</div>
""", unsafe_allow_html=True)

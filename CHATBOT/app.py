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

/* Main Background */
.stApp {
    background: #EAF1F8;
    color: #172B4D;
}

/* Main Container */
.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header */
.hero {
    background: linear-gradient(
        120deg,
        #12345A,
        #176B9A
    );
    padding: 35px 25px;
    border-radius: 18px;
    color: #FFFFFF;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: 0 5px 15px rgba(20, 50, 80, 0.15);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 30px;
    font-weight: 700;
}

.hero p {
    color: #EAF5FF !important;
    font-size: 15px;
}

/* Welcome Banner */
.gov-banner {
    background: #FFFFFF;
    border-radius: 15px;
    padding: 22px;
    text-align: center;
    margin: 20px 0;
    border: 1px solid #D5E0EC;
}

.gov-banner h3 {
    color: #12345A !important;
}

.gov-banner p {
    color: #40566F !important;
}

/* Feature Cards */
.feature {
    background: #FFFFFF;
    border: 1px solid #D5E0EC;
    padding: 18px 10px;
    border-radius: 13px;
    text-align: center;
    box-shadow: 0 3px 10px rgba(20, 50, 80, 0.07);
}

.feature h3 {
    color: #1479A8 !important;
}

.feature b {
    color: #173B60 !important;
    font-size: 14px;
}

/* General Text */
.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp div {
    color: #243B53;
}

/* Caption */
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #526579 !important;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #D9E3ED;
    border-radius: 13px;
    padding: 12px;
    margin-bottom: 12px;
}

/* Chat Message Text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: #172B4D !important;
}

/* Chat Input Area */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 2px solid #A9C5DA !important;
    border-radius: 12px !important;
}

/* Chat Input Text */
[data-testid="stChatInput"] textarea {
    color: #172B4D !important;
    background: #FFFFFF !important;
    caret-color: #176B9A !important;
}

/* Input Placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #718096 !important;
}

/* Buttons */
.stButton button {
    background: #176B9A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600;
    padding: 8px 20px;
}

.stButton button:hover {
    background: #12345A !important;
    color: #FFFFFF !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #526579 !important;
    padding: 20px;
    font-size: 13px;
    margin-top: 20px;
    border-top: 1px solid #D5E0EC;
}

/* Spinner */
.stSpinner p {
    color: #173B60 !important;
}

/* Links */
.stApp a {
    color: #086B9B !important;
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


# Welcome Section
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

    if not mykey:
        raise ValueError("API key is empty")

except Exception:

    st.error(
        "GROQ_API_KEY is missing. "
        "Please add your Groq API key in Streamlit Cloud Secrets."
    )

    st.stop()


# Load RAG System
@st.cache_resource
def initiate_rag():

    return load_base()


try:

    with st.spinner(
        "Loading government PDF documents..."
    ):

        index, embed_model, chunks = initiate_rag()

except Exception as e:

    st.error(
        f"Unable to load PDF documents: {str(e)}"
    )

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

        with st.spinner(
            "Searching documents and generating your answer..."
        ):

            try:

                answer = ask_rag(
                    question=question,
                    embed_model=embed_model,
                    chunks=chunks,
                    index=index,
                    api_key=mykey,
                    num_chunks=5,
                    model_name="openai/gpt-oss-120b"
                )

                st.markdown(answer)

                st.session_state.chat_hist.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                error_message = str(e)

                st.error(
                    f"Unable to generate answer: {error_message}"
                )

                st.session_state.chat_hist.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Sorry, I could not generate an answer. "
                            f"Error: {error_message}"
                        )
                    }
                )


# Footer
st.markdown("""
<div class="footer">
    Government Schemes & Policies Assistant | Powered by RAG
</div>
""", unsafe_allow_html=True)

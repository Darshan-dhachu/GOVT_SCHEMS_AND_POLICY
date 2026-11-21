import streamlit as st
from RAG_engine import load_base, ask_rag

st.set_page_config(page_title="Government Schemes & Policies", layout="centered")

# CSS Design
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, #DCEBFF 0%, transparent 40%),
        radial-gradient(circle at bottom right, #E8DEFF 0%, transparent 40%),
        linear-gradient(135deg, #F7F8FF, #EEF4FF, #F5F0FF);
    background-attachment: fixed;
    color: #102B46;
}
.block-container {max-width:950px;padding-top:2rem;padding-bottom:2rem;}
h1,h2,h3,p,label,span {color:#102B46 !important;}

.header {
    background:linear-gradient(135deg,#DCE8FA,#C9D9F5);
    padding:25px;border-radius:15px;text-align:center;
    margin-bottom:25px;border:1px solid #D1DDF0;
}
.header h1 {color:#102B46 !important;font-size:32px;font-weight:700;}

.welcome,.feature-card {
    background:#FFFFFF;padding:20px;border-radius:12px;
    border:1px solid #D5E2EF;
    box-shadow:0 4px 15px rgba(50,70,110,0.06);
}
.welcome {margin-bottom:20px;}
.feature-card {text-align:center;min-height:140px;padding:18px;}
.feature-icon {font-size:32px;margin-bottom:8px;}
.feature-card h4 {color:#172B4D !important;}

.stChatMessage {
    background:#FFFFFF;border-radius:12px;
    border:1px solid #D5E2EF;
}

/* Ask input text only */
.stChatInput textarea {
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
    caret-color:#FFFFFF !important;
}

/* Table dark text without background */
.stChatMessage table {
    background:transparent !important;
    border-collapse:collapse !important;
    width:100% !important;
}
.stChatMessage table th,
.stChatMessage table td {
    color:#102B46 !important;
    background:transparent !important;
    border:1px solid #B8C9DD !important;
    padding:10px !important;
}
.stChatMessage table th *,
.stChatMessage table td * {color:#102B46 !important;}

.stButton button {
    background:#1D5D8F;color:white;
    border-radius:8px;border:none;
}
.stButton button:hover {background:#16466D;color:white;}

.footer {
    text-align:center;color:#40566D;
    padding-top:25px;font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>Government Schemes & Policies</h1>
</div>
<div class="welcome">
    <h3>Welcome to Government Information Assistant</h3>
    <p>Get information about Indian Government schemes,
    policies and citizen benefits.</p>
</div>
""", unsafe_allow_html=True)

# Feature Cards
cards = [
    ("🏦", "Government Schemes", "Explore available schemes"),
    ("⚖️", "Government Policies", "Understand government policies"),
    ("🌟", "Citizen Benefits", "Know your available benefits")
]

cols = st.columns(3)
for col, (icon, title, desc) in zip(cols, cards):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Developed by DARSHAN")

# API Key
mykey = st.secrets["GROQ_API_KEY"]

# Load RAG only once
@st.cache_resource
def initiate_rag():
    return load_base()

try:
    with st.spinner("Loading government documents..."):
        index, embed_model, chunks = initiate_rag()
except Exception as e:
    st.error(f"Unable to load PDF documents: {e}")
    st.stop()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ask Question
question = st.chat_input("Ask about government schemes and policies...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding the answer..."):
            try:
                answer = ask_rag(
                    question, embed_model, chunks, index,
                    api_key=mykey,
                    num_chunks=5,
                    model_name="openai/gpt-oss-120b"
                )
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"Error generating answer: {e}")

# Footer
st.markdown("""
<div class="footer">
    Government Schemes & Policies | Powered by RAG
</div>
""", unsafe_allow_html=True)

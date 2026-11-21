import streamlit as st
from RAG_engine import load_base, ask_rag

st.set_page_config(
    page_title="Government Schemes & Policies",
    layout="centered"
)

# Custom Design
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, #DDD0FF 0%, transparent 35%),
        radial-gradient(circle at bottom right, #DDD0FF 0%, transparent 35%),
        linear-gradient(135deg, #F8F4FF, #EEE7FC);
    background-attachment: fixed;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, p, label, .stMarkdown {
    color: #17123F;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #17123F;
    margin-bottom: 12px;
}

.welcome-box {
    text-align: center;
    background: #E4D8FF;
    color: #17123F;
    padding: 12px;
    border-radius: 30px;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 12px;
}

.description {
    text-align: center;
    font-size: 17px;
    color: #27204D;
    margin-bottom: 25px;
}

.feature-card {
    background: rgba(226, 215, 255, 0.75);
    border: 2px solid #9270E8;
    border-radius: 16px;
    min-height: 145px;
    height: 145px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 12px;
}

.feature-icon {
    font-size: 48px;
    margin-bottom: 8px;
}

.feature-title {
    font-size: 18px;
    font-weight: 750;
    color: #17123F;
}

.developer {
    text-align: center;
    font-size: 17px;
    font-style: italic;
    color: #29205F;
    margin-top: 20px;
    margin-bottom: 20px;
}

hr {
    border: none;
    border-top: 2px solid #9270E8;
}

.stButton > button {
    background: #7650D5;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #5E3CB5;
    color: white;
}

[data-testid="stChatInput"] {
    border: 2px solid #D7C8FA;
    border-radius: 18px;
    background: #FFFFFF;
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.65);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# Main Heading
st.markdown(
    '<div class="main-title">Government Schemes & Policies</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="welcome-box">Welcome to Government Information Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">Get information about Indian Government schemes,<br> policies and citizen benefits.</div>',
    unsafe_allow_html=True
)

st.divider()

# Information Boxes
cards = [
    ("🏦", "Government Schemes"),
    ("⚖️", "Government Policies"),
    ("🌟", "Citizen Benefits")
]

cols = st.columns(3)

for col, (logo, title) in zip(cols, cards):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{logo}</div>
                <div class="feature-title">{title}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Developer Name
st.markdown(
    '<div class="developer">Developed by DARSHAN</div>',
    unsafe_allow_html=True
)

st.divider()

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

# Clear Chat
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display Previous Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ask Question
question = st.chat_input(
    "Ask about government schemes and policies..."
)

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Finding the answer..."):

            try:
                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
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
st.divider()
st.markdown(
    '<div style="text-align:center; color:#29205F;">Government Schemes & Policies</div>',
    unsafe_allow_html=True
)

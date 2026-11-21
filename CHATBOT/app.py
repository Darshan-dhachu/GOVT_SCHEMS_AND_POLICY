
import streamlit as st
from RAG_engine import load_base, ask_rag

st.set_page_config(
    page_title="Government Schemes & Policies",
    page_icon="🏛️",
    layout="wide"
)

# Header
st.title("🏛️ Government Schemes & Policies")
st.caption("Your information portal for Indian Government schemes and policies")
st.divider()

# Welcome Section
st.subheader("Welcome to Government Information Assistant")
st.write(
    "Explore government schemes, understand policies and discover "
    "benefits available to citizens across India."
)

st.divider()

# Information Cards
st.subheader("Explore Information")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("🏦")
        st.markdown("**Government Schemes**")
        st.caption("Explore schemes and initiatives.")

with col2:
    with st.container(border=True):
        st.subheader("⚖️")
        st.markdown("**Government Policies**")
        st.caption("Understand important policies.")

with col3:
    with st.container(border=True):
        st.subheader("🌟")
        st.markdown("**Citizen Benefits**")
        st.caption("Discover available benefits.")

st.divider()

# Developer Information
st.caption("Developed by DARSHAN")

# API Key
mykey = st.secrets["GROQ_API_KEY"]

# Load RAG
@st.cache_resource
def initiate_rag():
    return load_base()

try:
    with st.spinner("Loading government documents..."):
        index, embed_model, chunks = initiate_rag()
except Exception as e:
    st.error(f"Unable to load PDF documents: {e}")
    st.stop()

# Chat Section
st.subheader("💬 Ask Your Question")
st.write("Ask anything about government schemes and policies.")

if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([5, 1])

with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ask Question
question = st.chat_input("Ask about government schemes and policies...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

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

st.divider()

# Footer
st.caption("🏛️ Government Schemes & Policies | Powered by RAG")

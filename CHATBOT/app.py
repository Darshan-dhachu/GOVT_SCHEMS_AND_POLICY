
import streamlit as st
from RAG_engine import load_base, ask_rag

st.set_page_config(
    page_title="Government Schemes & Policies",
    layout="centered"
)

# Page Heading
st.title("Government Schemes & Policies")
st.subheader("Welcome to Government Information Assistant")
st.write("Get information about Indian Government schemes, policies and citizen benefits.")

# Feature Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.info("🏦 Government Schemes\n\nExplore available schemes")

with col2:
    st.info("⚖️ Government Policies\n\nUnderstand government policies")

with col3:
    st.info("🌟 Citizen Benefits\n\nKnow your available benefits")

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

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear Chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display Previous Messages
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

# Footer
st.caption("Government Schemes & Policies | Powered by RAG")

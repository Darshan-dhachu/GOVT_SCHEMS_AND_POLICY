import streamlit as st
from RAG_engine import load_base, ask_rag

st.set_page_config(
    page_title="Government Schemes & Policies",
    layout="centered"
)

# Main Heading
st.title("Government Schemes & Policies")

st.subheader("Welcome to Government Information Assistant")

st.write(
    "Get information about Indian Government schemes, "
    "policies and citizen benefits."
)

st.divider()

# Feature Cards
cards = [
    ("🏦", "Government Schemes", "Explore available schemes"),
    ("⚖️", "Government Policies", "Understand government policies"),
    ("🌟", "Citizen Benefits", "Know your available benefits")
]

cols = st.columns(3)

for col, (icon, title, desc) in zip(cols, cards):
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)

st.caption("Developed by DARSHAN")

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
if st.button("Clear Chat"):
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
st.caption("Government Schemes & Policies")

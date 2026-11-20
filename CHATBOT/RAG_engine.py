
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Load PDFs
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    mypages = [i.extract_text() or "" for i in reader.pages]
    full_text = "\n".join(mypages).strip()

    if not full_text:
        raise ValueError("No readable Text Found")

    return full_text


# Chunking
def create_chunks(full_text, chunksize=500, overlap=100):
    if chunksize <= 0 or overlap < 0 or overlap >= chunksize:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunksize
        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunksize - overlap

    return chunks


# Embedding
def build_vectors(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if not chunks:
        raise ValueError("Cannot apply embedding without chunks")

    embed_model = SentenceTransformer(model_name)
    embed_vectors = embed_model.encode(chunks, convert_to_numpy=True)
    embed_vectors = np.asarray(embed_vectors, dtype="float32")

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)

    return index, embed_model


# Load PDF, chunk and embedding
def load_base(pdf_path="Government_docs"):
    chunks = []
    pdf_files = list(Path(pdf_path).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in Government_docs folder")

    for pdf in pdf_files:
        full_text = load_pdf(pdf)
        chunks.extend(create_chunks(full_text))

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# Retrieve
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):
    if not question.strip():
        raise ValueError("Question cannot be blank")

    if not chunks:
        raise ValueError("No document chunks available")

    if num_chunks < 1:
        raise ValueError("Request at least one chunk")

    query_embed = embed_model.encode([question], convert_to_numpy=True)
    query_embed = np.asarray(query_embed, dtype="float32")

    faiss.normalize_L2(query_embed)

    sim_scores, vec_indices = index.search(
        query_embed, min(num_chunks, len(chunks))
    )

    background = ""

    for i in vec_indices[0]:
        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# Ask
def ask_rag(question, embed_model, chunks, index, api_key, num_chunks=3,
            model_name="openai/gpt-oss-120b"):

    if not api_key:
        raise ValueError("API key Missing")

    background = retrieve_content(
        question, embed_model, chunks, index, num_chunks
    )

    if not background.strip():
        return "Information missing from the document."

    prompt = """You are a professional Government Schemes and Policies AI Assistant.

Answer the user's question using only the provided document background.
Give simple, clear and accurate answers.
Do not invent or assume information.
If the answer is not available in the documents, say:
"Information missing from the document."

Document background:
""" + background + """

User question:
""" + question + """

Answer:"""

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

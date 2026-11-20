
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path


# Load All PDFs
def load_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))

    mypages = [page.extract_text() or "" for page in reader.pages]

    full_text = "\n".join(mypages).strip()

    if not full_text:
        raise ValueError(f"No readable text found in {pdf_path.name}")

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
        raise ValueError("No text available for embedding")

    embed_model = SentenceTransformer(model_name)

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embed_vectors = np.asarray(embed_vectors, dtype="float32")

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)

    return index, embed_model


# Load All PDFs, Chunk and Embedding
def load_base(pdf_path="Government_docs"):

    chunks = []

    folder_path = Path(pdf_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {pdf_path}")

    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in Government_docs")

    for pdf in pdf_files:

        full_text = load_pdf(pdf)

        pdf_chunks = create_chunks(full_text)

        chunks.extend(pdf_chunks)

    if not chunks:
        raise ValueError("No readable text found in PDF documents")

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# Retrieve
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):

    if not question.strip():
        raise ValueError("Question cannot be blank")

    if not chunks:
        raise ValueError("No document chunks available")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(query_embed, dtype="float32")

    faiss.normalize_L2(query_embed)

    sim_scores, vec_indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in vec_indices[0]:
        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# Ask
def ask_rag(question, embed_model, chunks, index, api_key,
            num_chunks=3, model_name="openai/gpt-oss-120b"):

    if not api_key:
        raise ValueError("API key Missing")

    background = retrieve_content(
        question, embed_model, chunks, index, num_chunks
    )

    if not background.strip():
        return "Information missing from the document."

    prompt = """
You are a professional Government Schemes and Policies AI Assistant.

Answer using only the provided document background.
Give simple, clear and accurate answers.
Use bullet points when needed.
Do not invent or assume facts.
If the answer is unavailable, say:
"Information missing from the document."

Document background:
""" + background + """

User question:
""" + question + """

Answer:
"""

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

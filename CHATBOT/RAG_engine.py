import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path


# Load PDF
def load_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))

    mypages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    full_text = "\n".join(mypages).strip()

    if not full_text:
        raise ValueError(
            f"No readable text found in {pdf_path.name}"
        )

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


# Embedding and FAISS
def build_vectors(chunks):

    if not chunks:
        raise ValueError("No chunks available for embedding")

    embed_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embed_vectors = np.asarray(
        embed_vectors,
        dtype="float32"
    )

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(
        embed_vectors.shape[1]
    )

    index.add(embed_vectors)

    return index, embed_model


# Load All PDFs
def load_base():

    chunks = []

    folder_path = Path(__file__).resolve().parent

    pdf_files = sorted(
        folder_path.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found in {folder_path}"
        )

    print(f"Found {len(pdf_files)} PDF files")

    for pdf in pdf_files:

        print(f"Loading: {pdf.name}")

        full_text = load_pdf(pdf)

        pdf_chunks = create_chunks(full_text)

        chunks.extend(pdf_chunks)

        print(
            f"Loaded {pdf.name}: {len(pdf_chunks)} chunks"
        )

    if not chunks:

        raise ValueError("No readable PDF content found")

    print(f"Total chunks: {len(chunks)}")

    index, embed_model = build_vectors(chunks)

    print("FAISS index created successfully")

    return index, embed_model, chunks


# Retrieve Relevant Content
def retrieve_content(
    question,
    embed_model,
    chunks,
    index,
    num_chunks=5
):

    if not question.strip():

        raise ValueError("Question cannot be blank")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(
        query_embed,
        dtype="float32"
    )

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


# Generate Answer using Groq
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=5,
    model_name="openai/gpt-oss-120b"
):

    if not api_key:

        raise ValueError("Groq API key is missing")

    # Retrieve relevant content
    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    if not background.strip():

        return "Information missing from the document."

    # Prompt
    prompt = f"""
You are a professional Government Schemes and Policies AI Assistant.

Answer the user's question using only the provided document content.

Instructions:

1. Give simple, clear and accurate answers.
2. Explain the scheme in easy language.
3. Include eligibility when available.
4. Include benefits when available.
5. Explain the application procedure when available.
6. Use pointwise answers where appropriate.
7. Do not invent information.
8. If the answer is unavailable, say:
Information missing from the document.

Document Background:

{background}

User Question:

{question}

Answer:
"""

    try:

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1500
        )

        answer = response.choices[0].message.content

        if not answer:

            raise ValueError("Groq returned an empty answer")

        return answer

    except Exception as e:

        raise RuntimeError(
            f"Groq API error: {str(e)}"
        ) from e

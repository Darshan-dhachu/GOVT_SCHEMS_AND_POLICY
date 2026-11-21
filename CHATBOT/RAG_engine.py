from pathlib import Path
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
# Read PDF Files
def load_pdf(pdf_path):
    full_text = ""
    pdf = PdfReader(pdf_path)
    for page in pdf.pages:
        text = page.extract_text() or ""
        full_text += text + "\n"
    if not full_text.strip():
        raise ValueError(
            f"No readable text found in {pdf_path.name}"
        )
    return full_text
# Create Text Chunks
def create_chunks(full_text, chunksize=700, overlap=150):
    chunks = []
    start = 0 
    while start < len(full_text):
        end = start + chunksize
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunksize - overlap
    return chunks
# Build Vector Database
def build_vectors(chunks):
    embed_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )
    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )
    index.add(embeddings)
    return index, embed_model
# Load All PDF Documents
def load_base():
    folder_path = Path(__file__).resolve().parent
    pdf_files = sorted(
        folder_path.glob("*.pdf")
    )
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {folder_path}"
        )
    all_chunks = []
    for pdf_file in pdf_files:
        text = load_pdf(pdf_file)
        chunks = create_chunks(text)
        all_chunks.extend(chunks)
    if not all_chunks:
        raise ValueError(
            "No text chunks created from PDF documents."
        )
    index, embed_model = build_vectors(
        all_chunks
    )
    return index, embed_model, all_chunks
# Retrieve Relevant Content
def retrieve_content(
    question,
    embed_model,
    chunks,
    index,
    num_chunks=5
):
    query_embedding = embed_model.encode(
        [question],
        convert_to_numpy=True
    )
    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(
        query_embedding,
        min(num_chunks, len(chunks))
    )
    relevant_chunks = []
    for i in indices[0]:
        if i >= 0:
            relevant_chunks.append(chunks[i])

    return "\n\n".join(relevant_chunks)


# Generate Answer Using Groq
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=5,
    model_name="openai/gpt-oss-120b"
):

    try:

        context = retrieve_content(
            question,
            embed_model,
            chunks,
            index,
            num_chunks
        )

        client = Groq(
            api_key=api_key,
            timeout=60.0,
            max_retries=2
        )

        prompt = f"""
You are a Government Schemes and Policies
Information Assistant for India.

Answer the user's question using the
provided government document context.

Instructions:
1. Give simple and clear answers.
2. Use headings and bullet points when useful.
3. Explain eligibility, benefits and application
   procedures when the context provides them.
4. Do not invent government scheme information.
5. If the answer is not available in the documents,
   clearly say that the information was not found
   in the uploaded documents.

Government Document Context:

{context}

User Question:

{question}

Answer:
"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=1500
        )

        return response.choices[0].message.content

    except Exception as e:

        raise RuntimeError(
            f"Groq answer generation failed: {e}"
        )

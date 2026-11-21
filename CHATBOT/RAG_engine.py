import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path


# Load PDF
def load_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError(
            f"No readable text found in {pdf_path.name}"
        )

    return full_text


# Create Chunks
def create_chunks(full_text, chunksize=700, overlap=150):

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


# Create Embeddings and FAISS Index
def build_vectors(chunks):

    if not chunks:
        raise ValueError("No text chunks available")

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


# Load All PDFs from CHATBOT Folder
def load_base():

    chunks = []

    folder_path = Path(__file__).resolve().parent

    pdf_files = sorted(
        folder_path.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found in: {folder_path}"
        )

    print(f"Found {len(pdf_files)} PDF files")

    for pdf in pdf_files:

        print(f"Loading PDF: {pdf.name}")

        try:

            full_text = load_pdf(pdf)

            pdf_chunks = create_chunks(full_text)

            chunks.extend(pdf_chunks)

            print(
                f"Loaded {pdf.name}: "
                f"{len(pdf_chunks)} chunks"
            )

        except Exception as e:

            raise ValueError(
                f"Error reading {pdf.name}: {e}"
            ) from e

    if not chunks:

        raise ValueError(
            "No readable content found in PDF files"
        )

    print(f"Total chunks created: {len(chunks)}")

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

    if not question or not question.strip():

        raise ValueError("Please enter a question")

    if not chunks:

        raise ValueError("PDF document chunks are empty")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(
        query_embed,
        dtype="float32"
    )

    faiss.normalize_L2(query_embed)

    scores, indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for score, i in zip(scores[0], indices[0]):

        if i != -1:

            background += chunks[i] + "\n\n"

    if not background.strip():

        raise ValueError(
            "No relevant information retrieved from PDFs"
        )

    print("Retrieved document content successfully")

    return background


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

    if not api_key:

        raise ValueError(
            "Groq API key is missing"
        )

    # Retrieve PDF Content
    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    # Create Prompt
    prompt = f"""
You are a helpful Indian Government Schemes and Policies Assistant.

Answer the user's question using the supplied PDF document content.

Instructions:
1. Understand the user's question carefully.
2. Give simple, clear and useful answers.
3. Use headings and bullet points when appropriate.
4. Explain eligibility, benefits and application procedures
   when this information is available.
5. Do not make up facts, dates, amounts or eligibility rules.
6. If the answer is not available in the supplied content,
   clearly say that the information is not available in the
   uploaded documents.
7. If the question is general, answer it only when the
   document content supports the answer.

PDF DOCUMENT CONTENT:

{background}

USER QUESTION:

{question}

ANSWER:
"""

    try:

        client = Groq(
            api_key=api_key,
            timeout=60.0,
            max_retries=2
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a reliable assistant. "
                        "Answer using the supplied document context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=1500
        )

        if not response.choices:

            raise ValueError(
                "Groq returned no response choices"
            )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():

            raise ValueError(
                "Groq returned an empty answer"
            )

        return answer.strip()

    except Exception as e:

        raise RuntimeError(
            f"Groq answer generation failed: {str(e)}"
        ) from e

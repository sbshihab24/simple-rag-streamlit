import os
import re
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from groq import Groq

# ========================
# Load Environment Variables
# ========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is missing. Add it to your .env or Streamlit secrets.")

# ========================
# Load Embedding Model (Cached)
# ========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ========================
# Sentence-Aware PDF Chunking
# ========================
def split_into_chunks(text, max_words=1000):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current_chunk = [], []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        if current_len + word_count <= max_words:
            current_chunk.append(sentence)
            current_len += word_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ========================
# Extract Text from PDF
# ========================
def extract_text_from_pdf(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# ========================
# Process PDF and Create Embeddings (Cached)
# ========================
@st.cache_data
def process_pdf_and_create_embeddings(pdf_bytes, _embedding_model):
    raw_text = extract_text_from_pdf(pdf_bytes)
    text_chunks = split_into_chunks(raw_text)
    embeddings = _embedding_model.encode(text_chunks, convert_to_tensor=True)
    return text_chunks, embeddings

# ========================
# Perform Semantic Search
# ========================
def perform_semantic_search(query, chunks, chunk_embeddings, embedding_model, top_k=3):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_results = similarities.topk(k=top_k)
    top_chunks = [chunks[i] for i in top_results.indices]
    return top_chunks

# ========================
# Generate Response from GROQ
# ========================
def generate_response(system_prompt, user_prompt):
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ========================
# Streamlit UI
# ========================
def main():
    st.set_page_config(page_title="PDF RAG App with GROQ", layout="wide")
    st.title("📄 PDF RAG App using SentenceTransformer + GROQ")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("🔍 Processing document and generating embeddings..."):
            embedding_model = load_embedding_model()
            text_chunks, chunk_embeddings = process_pdf_and_create_embeddings(uploaded_file.read(), embedding_model)

        st.success("✅ PDF processed and embedded!")

        query = st.text_input("🔎 Ask a question based on the PDF:")

        if query:
            with st.spinner("🧠 Searching and generating answer..."):
                top_chunks = perform_semantic_search(query, text_chunks, chunk_embeddings, embedding_model)
                user_prompt = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])
                user_prompt += f"\n\nQuestion: {query}"

                system_prompt = "You are a helpful assistant. Answer based on the context provided."
                answer = generate_response(system_prompt, user_prompt)

            st.subheader("📘 Answer:")
            st.write(answer)

            with st.expander("🧩 Top Context Chunks Used"):
                for i, chunk in enumerate(top_chunks):
                    st.markdown(f"**Chunk {i+1}:** {chunk}")

if __name__ == "__main__":
    main()

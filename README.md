# 📄 Simple RAG Streamlit App Using GROQ + SentenceTransformer

A clean, hands-on implementation of a **Retrieval-Augmented Generation (RAG)** system built from scratch using familiar Python tools:

- PDF text extraction with **PyMuPDF**
- Sentence-aware chunking and embeddings with **SentenceTransformer**
- Semantic search via cosine similarity
- GROQ LLM API for fast, grounded answer generation
- Streamlit app for effortless PDF QA interaction

---

## 🚀 What’s Inside?

This repo includes:

| File                   | Description                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------------|
| `rag_using_groq.ipynb` | Jupyter Notebook for prototyping RAG pipeline and experimentation. This is where core logic was developed, tested, and refined. |
| `app.py`               | Main Streamlit application wrapping the RAG pipeline into an interactive web interface. Upload PDFs, ask questions, get answers! |
| `.env`                 | Environment variables file — store your `GROQ_API_KEY` here securely.                              |
| `requirements.txt`     | Required Python packages to run the notebook and app.                                              |
| `README.md`            | This file — project overview, setup instructions, and usage.                                      |

---

## 🧠 How It Works

### 1. Prototype in Notebook: `rag_using_groq.ipynb`

- Extract text from PDFs with **PyMuPDF**
- Split text into meaningful sentence-aware chunks for semantic coherence
- Generate chunk embeddings using the `all-MiniLM-L6-v2` SentenceTransformer model
- Perform semantic search with cosine similarity to identify relevant chunks per query
- Call GROQ LLM API with retrieved context for accurate, grounded responses
- Tune chunk sizes, overlap, and prompt engineering for best performance

### 2. Productionize in Streamlit: `app.py`

- User uploads a PDF document
- App extracts and chunks the text, caching embeddings for efficiency
- User asks questions via the UI
- Semantic search finds top relevant chunks
- GROQ LLM generates answers based on those chunks
- UI shows answers and the context chunks used for transparency

---

## 🛠️ Installation & Setup

### 1. Clone this repository:

```bash
git clone https://github.com/sbshihab24/simple-rag-streamlit.git
cd simple-rag-streamlit
2. Install dependencies:



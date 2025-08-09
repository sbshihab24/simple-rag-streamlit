# 📄 Simple RAG Streamlit App Using GROQ + SentenceTransformer



A clean, hands-on implementation of a **Retrieval-Augmented Generation (RAG)** system built from scratch using familiar Python tools:

- PDF text extraction with PyMuPDF
- Sentence-aware chunking and embeddings with SentenceTransformer
- Semantic search via cosine similarity
- GROQ LLM API for fast, grounded answer generation
- Streamlit app for effortless PDF QA interaction

---

## 🚀 What’s Inside?

This repo includes:

| File                | Description                                                   |
|---------------------|---------------------------------------------------------------|
| `rag_using_groq.ipynb` | Jupyter Notebook for prototyping RAG pipeline and experimentation. This is where core logic was developed, tested, and refined. |
| `app.py`            | Main Streamlit application wrapping the RAG pipeline into an interactive web interface. Upload PDFs, ask questions, get answers! |
| `.env`              | Environment variables file — store your `GROQ_API_KEY` here securely. |
| `requirements.txt`  | Required Python packages to run the notebook and app.         |
| `README.md`         | This file — project overview, setup instructions, and usage.  |

---

## 🧠 How It Works

### 1. Prototype in Notebook: `rag_using_groq.ipynb`

- Extract text from PDFs with PyMuPDF
- Split text into meaningful sentence-aware chunks for semantic coherence
- Generate chunk embeddings using `all-MiniLM-L6-v2` SentenceTransformer model
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
### 2. Install dependencies:
pip install -r requirements.txt
### 3. Set your GROQ API Key:
Create a .env file in the root directory with:
GROQ_API_KEY=your_groq_api_key_here
## ⚡ Run the App
streamlit run app.py

## 🔍 Core Concepts
Sentence-Aware Chunking: Keeps text chunks semantically complete to preserve meaning

Embeddings: Converts text chunks into numerical vectors with SentenceTransformer

Semantic Search: Uses cosine similarity to find chunks most relevant to your query

GROQ LLM API: Answers questions grounded on the retrieved context for accuracy and reliability

Caching: Efficient loading and reuse of embedding models and chunk embeddings for performance

Streamlit UI: User-friendly interface for easy PDF-based Q&A without coding

## 🗂️ Project Structure Overview
simple-rag-streamlit/
├── app.py                 # Streamlit app for PDF upload & Q&A
├── rag_using_groq.ipynb   # Notebook prototyping the RAG pipeline
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (GROQ_API_KEY)
├── README.md              # This file
└── data/
    └── val.json                       <- Sample validation data (queries and answers)
    └── AI_information.pdf             <- A sample PDF document for testing.
    └── attention_is_all_you_need.pdf  <- A sample PDF document for testing (for Multi-Modal RAG).

### 🤝 Contributions
Contributions, issues, and feature requests are welcome! Feel free to open a PR or submit an issue.

If you find this project helpful, please ⭐ star the repo!

### 📫 Contact
Mehedi Hasan Shihab

GitHub: sbshihab24

LinkedIn: shihab24

Email: sbshihab2000@gmail.com

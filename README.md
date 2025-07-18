# GenAI_Projects

## Project 1: 🧬 Curie – Scientific Research AI Assistant
Curie is an AI-powered research assistant built using Streamlit, designed specifically for researchers, medical professionals, and data scientists working with medical and pharmaceutical PDF documents.

This smart app lets you:
- ✅ Summarize lengthy scientific PDFs
- ✅ Find similar PDFs online (via Google Search)
- ✅ Chat with your uploaded PDF (using RAG – Retrieval Augmented Generation)

## 🚀 Features:
### 📄 Upload & Parse PDFs
- Upload any scientific or regulatory PDF (clinical trials, FDA documents, research papers).
- Extracts full text using **PyMuPDF** for high-quality parsing.
### 📝 AI-Powered Summarization
- Generates concise summaries using `distilbart-cnn-12-6` transformer model.
- Downloadable summary as `.txt` file.
### 🔍 Similar PDF Recommendation
- Finds 5–6 similar PDFs from Google Search.
- Based on document summary or fallback to intro text.
- Useful for comparative analysis and literature review.
### 💬 Chat with PDF (RAG-powered)
- Ask natural language questions like:
  - "What are the side effects?"
  - "Key findings?"
  - "What is the trial design?"
- Uses:
  - **FAISS** similarity search over semantic chunks
  - **MiniLM-L6-v2** sentence embeddings
  - **RoBERTa SQuAD2** QA model

### 🧠 Persistent User State
- `st.session_state` preserves:
  - Chat history
  - PDF recommendations
  - Prevents UI flicker or blur

## 🛠️ Tech Stack
| Layer            | Tools                                   |
|------------------|------------------------------------------|
| App Framework    | [Streamlit](https://streamlit.io)        |
| PDF Text Parsing | PyMuPDF (`fitz`)                         |
| Summarization    | `distilbart-cnn-12-6` (transformers)     |
| Embedding Model  | `sentence-transformers/MiniLM-L6-v2`     |
| Search Index     | `FAISS`                                  |
| Question Answer  | `deepset/roberta-base-squad2`            |
| Web Search       | `googlesearch-python`                    |

## 📂 How to Run Locally
```bash
git clone https://github.com/your-username/curie-medical-research-bot.git
cd curie-medical-research-bot
pip install -r requirements.txt
streamlit run app.py
```

## UI
<img width="800" height="373" alt="image" src="https://github.com/user-attachments/assets/02619c71-9d8c-4e13-828d-5e47b507ca5e" />


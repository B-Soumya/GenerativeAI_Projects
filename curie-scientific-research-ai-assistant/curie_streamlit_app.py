import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import hashlib
import base64
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from io import StringIO
from googlesearch import search  # ✅ Google search for similar PDFs

# ----------------- App Setup -----------------
st.set_page_config("🧬 Curie", layout="wide")
st.title("🧬 Curie - A Scietific Research Bot")
st.markdown("**Let's Summarize. Get Recommendation. Chat with the document.**")
st.markdown("**All from your Medical and Pharma PDFs.**")

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    reader = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embedder, summarizer, reader

embedder, summarizer, reader = load_models()

# ----------------- Utility Functions -----------------
def extract_text_from_pdf_bytes(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return " ".join(page.get_text() for page in doc).strip()

def summarize_text(text, max_lines=15):
    text = text.replace("\n", " ").strip()
    chunks = [text[i:i+800] for i in range(0, len(text), 800)]
    summaries = []
    for chunk in chunks:
        if len(chunk) < 50:
            continue
        summary = summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
        if len(summaries) >= max_lines:
            break
    combined = " ".join(summaries)
    lines = combined.split('. ')
    return ". ".join(lines[:max_lines]) + '.' if lines else "No summary available."

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    if not chunks:
        return None, None
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def get_top_chunks(query, chunks, index, k=3):
    if not chunks or index is None:
        return ["No content available."]
    k = min(k, len(chunks))
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

def answer_with_rag(question, chunks):
    context = " ".join(chunks)
    if len(context.strip()) == 0:
        return "No answer found."
    try:
        result = reader(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"⚠️ Error: {e}"

def get_text_download_link(text, filename):
    buffer = StringIO()
    buffer.write(text)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">💾 Download Summary (.txt)</a>'
    return href

def get_file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()

def recommend_similar_pdfs(summary, num_results=5):
    query = summary.strip().split(".")[0][:120] + " filetype:pdf"
    try:
        results = list(search(query, num_results=num_results))
        return [url for url in results if url.endswith(".pdf") or "pdf" in url]
    except Exception as e:
        return [f"❌ Error fetching results: {e}"]

# ----------------- Sidebar Upload -----------------
with st.sidebar:
    st.header("📎 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# ----------------- Main Application -----------------
if uploaded_file:
    file_hash = get_file_hash(uploaded_file)

    # Auto-reset on new file
    if "last_file_hash" not in st.session_state or st.session_state.last_file_hash != file_hash:
        st.session_state.clear()
        st.session_state.last_file_hash = file_hash
        st.session_state.file_bytes = uploaded_file.read()
        st.session_state.summary = ""
        st.session_state.chat_history = []

    if "pdf_text" not in st.session_state:
        with st.spinner("📖 Extracting text from PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf_bytes(st.session_state.file_bytes)

    if "chunks" not in st.session_state or "index" not in st.session_state:
        with st.spinner("🔗 Preparing for Q&A..."):
            chunks = chunk_text(st.session_state.pdf_text)
            index, _ = embed_chunks(chunks)
            st.session_state.chunks = chunks
            st.session_state.index = index
    
    
    # ----------------- Summary Section -----------------
    st.subheader("📚 PDF Summary")

    if st.button("Get Summary"):
        with st.spinner("📝 Generating summary..."):
            st.session_state.summary = summarize_text(st.session_state.pdf_text)

    if "summary" in st.session_state and st.session_state.summary:
        summary_text = st.session_state.summary
        st.markdown("""
        <div style='padding: 1rem; background-color: #f8f9fa; border: 1px solid #dee2e6;
                    border-radius: 10px; text-align: justify; font-size: 16px; line-height: 1.6;'>
        {}</div>
        """.format(summary_text.replace("\n", "<br>")), unsafe_allow_html=True)

        st.markdown(get_text_download_link(summary_text, "pdf_summary.txt"), unsafe_allow_html=True)
    else:
        st.info("Click **📄 Get Summary** to generate a detailed summary.")

    
    # ----------------- PDF Recommendation -----------------
    st.markdown("---")
    st.subheader("🔍 Recommend Similar PDFs")

    if st.button("🔗 Find Similar PDFs"):
        with st.spinner("🌐 Searching for similar PDFs..."):
            # Use summary if available; fallback to first 1000 characters of PDF
            recommendation_basis = (
                st.session_state.get("summary", "") or st.session_state.pdf_text[:1000]
            )
            urls = recommend_similar_pdfs(recommendation_basis)
            st.session_state.recommended_urls = urls

    # Always display recommended URLs if they exist
    if "recommended_urls" in st.session_state:
        urls = st.session_state.recommended_urls
        if urls and all(url.startswith("http") for url in urls):
            st.markdown("🌐 **Top 5 Similar PDFs from Trusted Sources:**")
            for idx, url in enumerate(urls, 1):
                display_text = url.split("/")[-1] if len(url.split("/")) > 2 else f"Similar PDF {idx}"
                st.markdown(f"{idx}. [🔗 {display_text}]({url})")
        else:
            st.warning("⚠️ No similar PDFs found or search failed.")
    else:
        st.info("Click the **🔗 Find Similar PDFs** button to fetch recommendations.")
    

    # ----------------- Chat Section -----------------
    st.markdown("---")
    st.subheader("💬 Chat with PDF (RAG-powered)")

    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input(
            "Ask a question about the PDF:",
            placeholder="e.g., What are the key findings?",
            key="chat_input"
        )
        submitted = st.form_submit_button("Ask")

    if submitted and query and st.session_state.index:
        with st.spinner("🤖 Thinking..."):
            top_chunks = get_top_chunks(query, st.session_state.chunks, st.session_state.index)
            
            if not top_chunks:
                response = "⚠️ Sorry, I couldn't find relevant information in the PDF."
            else:
                context = " ".join(top_chunks)
                try:
                    result = reader(question=query, context=context)
                    response = result.get('answer', '').strip()
                    if not response:
                        response = "🤔 I couldn't find a confident answer. Try rephrasing your question."
                except Exception as e:
                    response = f"⚠️ Error from model: {e}"

            st.session_state.chat_history.append(("🧑‍💻 You", query))
            st.session_state.chat_history.append(("📄 Curie", response))

    # Display full chat history
    if "chat_history" in st.session_state:
        for speaker, msg in st.session_state.chat_history[::-1]:
            st.markdown(f"**{speaker}:** {msg}")



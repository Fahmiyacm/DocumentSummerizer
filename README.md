# 📄 Business Document Intelligence Platform

An AI-powered **Business Document Intelligence Platform** that allows users to upload business documents and perform **summarization, document-based Q&A, and risk analysis**.

## 🚀 Features

* 📄 Upload **PDF, DOCX, and TXT** documents
* 📝 Generate AI-powered document summaries
* 💬 Ask questions about the uploaded document using **RAG**
* 🔍 Retrieve relevant document chunks using **FAISS**
* ⚠️ Identify potential business risks and important clauses
* 📌 Display document sources and page numbers

## 🛠️ Technologies

* **Python**
* **Streamlit** – Web interface
* **LangChain** – Document processing and RAG
* **Hugging Face Embeddings** – Text embeddings
* **FAISS** – Vector similarity search
* **LLM** – Answer generation, summarization, and risk analysis
* **PyPDF / python-docx** – Document extraction

## 📂 Project Structure

```text
Business-Document-Intelligence/
│
├── app.py
├── src/
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embedding.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── summarizer.py
│   └── risk_analyzer.py
│
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open the URL displayed by Streamlit in your browser.

## 🔄 Workflow

```text
Upload Document
       ↓
Extract Text
       ↓
Split into Chunks
       ↓
Generate Embeddings
       ↓
Store in FAISS
       ↓
 ┌───────────────┬───────────────┬───────────────┐
 ↓               ↓               ↓
Summary          Q&A        Risk Analysis
```

## 🎯 Use Cases

This platform can be used for analyzing:

* Business contracts
* Rental agreements
* Company policies
* Legal/business documents
* Terms and conditions
* Other structured business documents

## ⚠️ Disclaimer

The risk analysis feature provides **AI-generated insights** and should not be considered a substitute for professional legal or business advice.

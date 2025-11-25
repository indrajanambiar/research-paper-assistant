# 📚 Research Paper Assistant (RAG)

A powerful RAG (Retrieval-Augmented Generation) application designed to help researchers interact with and extract insights from PDF research papers. Built with **Streamlit**, **LangChain**, and **Groq (Llama 3.1)** for blazing fast and accurate responses.

## 🚀 Key Features

*   **⚡ Blazing Fast Inference**: Uses **Groq API** with Llama 3.1-8b-instant for near-instant answers.
*   **📄 PDF Ingestion**: Upload multiple research papers (PDFs) to build your knowledge base.
*   **🔍 Smart Retrieval**: Uses semantic search (ChromaDB + SentenceTransformers) to find relevant context.
*   **🎯 Session-Based Querying**: Upload a file and query *only* that file instantly, without distraction from the rest of the database.
*   **🧹 Auto-Deduplication**: Automatically cleans up old versions of a file when you re-upload it, keeping your database clean.
*   **🗑️ Easy Management**: View and delete documents from your knowledge base via the UI.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **LLM Engine**: Groq API (Llama 3.1-8b-instant)
*   **Vector Store**: ChromaDB
*   **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
*   **Orchestration**: LangChain (Text Splitting & Vector Store Management)

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd ResearchPaperKnowledgeBase
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your Groq API Key:
    ```env
    GROQ_API_KEY=gsk_your_actual_api_key_here
    ```
    *Get a free key at [console.groq.com](https://console.groq.com).*

## 🏃‍♂️ Usage

1.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

2.  **Upload Papers**:
    *   Go to the **"Upload Documents"** tab.
    *   Drag and drop your PDF files.
    *   Click "Process Files" to ingest them into the vector database.

3.  **Chat & Query**:
    *   Go to the **"Chat & Query"** tab.
    *   Ask questions like:
        *   *"Summarize the paper 'Attention is All You Need'"*
        *   *"What are the key findings regarding OCR VQGAN?"*
        *   *"Compare the methodology of the two uploaded papers"*
    *   **Session Filter**: If you just uploaded files, the chat will focus on *those specific files*. Click "Clear filter" to search the entire database.

## 📂 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── src/
│   ├── config.py          # Configuration settings
│   ├── ingest.py          # PDF text extraction and processing
│   ├── llm.py             # Groq LLM integration
│   ├── rag.py             # RAG pipeline logic
│   ├── vector_store.py    # ChromaDB management
│   └── prompts.py         # System prompts
├── data/                  # Directory for storing raw PDFs
├── chroma_db/             # Persistent vector database storage
├── requirements.txt       # Python dependencies
└── .env                   # API keys (not committed)
```

## ⚠️ Troubleshooting

*   **"Model not found"**: We switched to Groq, so local model files are no longer needed. Ensure your `GROQ_API_KEY` is set.
*   **"I cannot answer this..."**: The model is strict about using only provided context. If the answer isn't in the retrieved chunks, it will say so. Try increasing the number of retrieved chunks or rephrasing.
*   **Slow Uploads**: Large PDFs take time to embed (CPU-bound). Please be patient.

## 📜 License

MIT License

# 💬 ByteBot - RAG Customer Support Assistant

A Retrieval-Augmented Generation (RAG) customer support system built with Python, Streamlit, ChromaDB, and the DeepSeek API. ByteBot answers customer inquiries regarding company policies, warranties, and shipping details using strictly retrieved context to prevent hallucinations.

## 🤖 System Architecture & Technical Stack
* **Frontend Interface:** Streamlit (Interactive web UI with citation expanders)
* **Vector Store & Indexing:** ChromaDB (Local vector storage)
* **Embedding & Retrieval:** Scikit-Learn `TfidfVectorizer` (TF-IDF vector representation)
* **LLM Engine:** DeepSeek Chat API (`deepseek-v4-flash` via OpenAI SDK)
* **Environment Configuration:** `python-dotenv` for local API key management

## 🔑 Key Features
* **Context-Grounded Answers:** Constrained by a system prompt to answer strictly using retrieved knowledge base documents.
* **Source Transparency:** Displays expandable source documents for every query directly in the UI.
* **Session State Management:** Preserves chat history across conversational turns.

## 🛠️ Project Structure
```text
percy-rag-bot/
├── app.py              # Streamlit application UI
├── llm_with_rag.py     # RAG pipeline, ChromaDB store, and LLM orchestration
├── requirements.txt    # Project dependencies
└── README.md           # Documentation

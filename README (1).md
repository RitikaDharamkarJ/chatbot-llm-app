# AI Chatbot with LangChain RAG

A production-ready conversational AI chatbot built with OpenAI, LangChain, and Streamlit. Features document-aware Q&A using Retrieval-Augmented Generation (RAG), multiple assistant personas, and real-time parameter tuning.

---

## Features

- **Conversational AI** — Multi-turn chat with memory using OpenAI GPT models
- **RAG (Document Q&A)** — Upload a `.txt` file and ask questions about it using LangChain + FAISS vector search
- **Prompt Engineering** — 4 switchable assistant personas (General, Research, Code, Creative)
- **Parameter Tuning** — Adjust temperature, max tokens, and memory window in real time
- **Session Metrics** — Tracks total queries and messages per session
- **Clean UI** — Dark-themed Streamlit interface with chat bubbles and status indicators

---

## Tech Stack

| Tool | Purpose |
|---|---|
| OpenAI GPT | Language model for chat completions |
| LangChain | Text splitting, embeddings, vector store |
| FAISS | Vector similarity search for RAG |
| Streamlit | Interactive web UI |
| Python 3.13 | Core language |

---

## Project Structure

```
chatbot-llm-app/
├── notebook/
│   └── chatbot_app.py      # Main application
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/RitikaDharamkarJ/chatbot-llm-app.git
cd chatbot-llm-app
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install streamlit openai langchain langchain-openai langchain-community faiss-cpu
```

### 4. Run the app

```bash
streamlit run notebook/chatbot_app.py
```

### 5. Open in browser

```
http://localhost:8501
```

---

## Usage

1. Enter your **OpenAI API key** in the sidebar
2. Select a **model** (default: `gpt-4o-mini`)
3. Choose an **assistant persona** based on your task
4. Adjust **temperature** and **max tokens** as needed
5. Type a message and click **Send**

### Document Q&A (RAG)
1. Upload a `.txt` file in the sidebar under **Document Q&A**
2. Once processed, ask any question about the document
3. The chatbot will search relevant sections and answer accordingly

---

## How RAG Works

```
User uploads .txt file
        ↓
LangChain splits into chunks       (RecursiveCharacterTextSplitter)
        ↓
Chunks embedded via OpenAI         (OpenAIEmbeddings)
        ↓
Stored in FAISS vector store
        ↓
User asks a question
        ↓
LangChain finds relevant chunks    (similarity_search)
        ↓
OpenAI answers with document context
```

---

## Assistant Personas

| Persona | Best For |
|---|---|
| General Assistant | Everyday questions and conversations |
| Research & Analysis | Data, research, and structured insights |
| Code & Technical | Programming help and technical explanations |
| Creative Writing | Stories, poems, and creative content |

---

## Requirements

- Python 3.9+
- OpenAI API key with available credits ([platform.openai.com](https://platform.openai.com))

---

## Author

**Ritika Dharamkar**
- GitHub: [@RitikaDharamkarJ](https://github.com/RitikaDharamkarJ)
- Email: ritikadharmkar2023@gmail.com

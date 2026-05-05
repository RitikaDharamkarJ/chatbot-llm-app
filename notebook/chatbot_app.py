import streamlit as st
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import tempfile
import os

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }
    .chat-header { text-align: center; padding: 2rem 0 1rem; }
    .chat-header h1 {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }
    .chat-header p { color: #8892a4; font-size: 0.95rem; margin-top: 0.4rem; }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] label {
        color: #a0aec0 !important;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .user-message { display: flex; justify-content: flex-end; margin: 0.75rem 0; }
    .user-bubble {
        background: linear-gradient(135deg, #7b2ff7, #00d2ff);
        color: white;
        padding: 0.85rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 72%;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 4px 20px rgba(123, 47, 247, 0.3);
    }

    .assistant-message {
        display: flex; justify-content: flex-start;
        margin: 0.75rem 0; align-items: flex-start; gap: 10px;
    }
    .assistant-avatar {
        width: 36px; height: 36px; border-radius: 50%;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
    }
    .assistant-bubble {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        color: #e2e8f0;
        padding: 0.85rem 1.2rem;
        border-radius: 4px 18px 18px 18px;
        max-width: 72%; font-size: 0.95rem; line-height: 1.7;
    }

    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.75rem;
    }
    .metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #00d2ff; }
    .metric-label { font-size: 0.75rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.08em; }

    .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #e2e8f0 !important; border-radius: 12px !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #7b2ff7, #00d2ff) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
    }
    .chat-container {
        max-height: 520px; overflow-y: auto; padding: 1rem;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; margin-bottom: 1rem;
    }
    .status-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 500;
        background: rgba(0,210,100,0.15); color: #00d264;
        border: 1px solid rgba(0,210,100,0.3);
    }
    .rag-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 500;
        background: rgba(123,47,247,0.15); color: #a78bfa;
        border: 1px solid rgba(123,47,247,0.3); margin-left: 8px;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# System Prompts (Prompt Engineering)
# ─────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "General Assistant": "You are a helpful, friendly, and knowledgeable AI assistant. Your responses are clear, concise, and accurate.",
    "Research & Analysis": "You are an expert research assistant. Break down problems systematically and provide structured, evidence-based insights.",
    "Code & Technical": "You are a senior software engineer. Provide clean, well-commented code and explain technical concepts clearly.",
    "Creative Writing": "You are a creative writing partner. Help users craft compelling stories, poems, and creative content."
}


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ─────────────────────────────────────────────
# LangChain RAG — Document Processing
# ─────────────────────────────────────────────
def process_document(uploaded_file, api_key):
    """
    Uses LangChain to:
    1. Split document into chunks
    2. Embed chunks using OpenAI Embeddings
    3. Store in FAISS vector store for similarity search
    """
    # Read file content
    content = uploaded_file.read().decode("utf-8", errors="ignore")

    # LangChain: split text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents([content])

    # LangChain: embed chunks + store in FAISS vector DB
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def get_relevant_context(vectorstore, query, k=3):
    """
    Uses LangChain FAISS to find the most relevant
    document chunks for the user's question
    """
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])


# ─────────────────────────────────────────────
# Core Chat Function — Direct OpenAI
# ─────────────────────────────────────────────
def get_response(api_key, model, temperature, max_tokens, persona, chat_history, user_message, context=None):
    client = OpenAI(api_key=api_key)

    system_prompt = SYSTEM_PROMPTS[persona]

    # If document context exists, inject it into the system prompt
    if context:
        system_prompt += f"""

You also have access to the following document context. Use it to answer questions accurately.
If the answer is in the document, reference it. If not, answer from your general knowledge.

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key from platform.openai.com"
    )

    model_choice = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"],
        index=0
    )

    persona = st.selectbox(
        "Assistant Persona",
        list(SYSTEM_PROMPTS.keys()),
        index=0
    )

    st.markdown("### 🎛️ Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05,
                            help="Higher = more creative, Lower = more focused")
    max_tokens = st.slider("Max Response Tokens", 256, 2048, 1024, 128)

    # ── RAG: Document Upload ──
    st.markdown("---")
    st.markdown("### 📄 Document Q&A (RAG)")
    st.caption("Upload a .txt file to chat with your document")

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["txt"],
        label_visibility="collapsed"
    )

    if uploaded_file and api_key:
        if uploaded_file.name != st.session_state.doc_name:
            with st.spinner("Processing document..."):
                try:
                    st.session_state.vectorstore = process_document(uploaded_file, api_key)
                    st.session_state.doc_name = uploaded_file.name
                    st.success(f"✅ '{uploaded_file.name}' ready!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    if st.session_state.doc_name:
        st.info(f"📄 Active: {st.session_state.doc_name}")
        if st.button("Remove Document", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.doc_name = None
            st.rerun()

    st.markdown("---")

    # Metrics
    st.markdown("### 📊 Session Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{st.session_state.total_queries}</div>
            <div class="metric-label">Queries</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(st.session_state.messages)}</div>
            <div class="metric-label">Messages</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.total_queries = 0
        st.rerun()


# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>🤖 AI Chatbot</h1>
    <p>Powered by OpenAI · LangChain RAG · Conversational Memory · Prompt Engineering</p>
</div>
""", unsafe_allow_html=True)

# Status badges
if api_key:
    badge_html = '<span class="status-badge">● Connected</span>'
    if st.session_state.doc_name:
        badge_html += '<span class="rag-badge">📄 RAG Active</span>'
    st.markdown(badge_html, unsafe_allow_html=True)
else:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start chatting.")

st.markdown("---")

# ── Chat History ──
chat_html = ""
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += f"""
        <div class="user-message">
            <div class="user-bubble">{msg["content"]}</div>
        </div>"""
    else:
        chat_html += f"""
        <div class="assistant-message">
            <div class="assistant-avatar">🤖</div>
            <div class="assistant-bubble">{msg["content"]}</div>
        </div>"""

st.markdown(f'<div class="chat-container">{chat_html}</div>', unsafe_allow_html=True)

# ── Input ──
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "Message",
        placeholder="Type your message here... (or ask about your uploaded document)",
        label_visibility="collapsed",
        key="user_input"
    )
with col_btn:
    send = st.button("Send →", use_container_width=True)


# ─────────────────────────────────────────────
# Handle Send
# ─────────────────────────────────────────────
if send and user_input.strip():
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.total_queries += 1

        with st.spinner("Thinking..."):
            try:
                # If document is uploaded, use LangChain RAG to get relevant context
                context = None
                if st.session_state.vectorstore:
                    context = get_relevant_context(st.session_state.vectorstore, user_input)

                # Get response from OpenAI (with or without document context)
                reply = get_response(
                    api_key=api_key,
                    model=model_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    persona=persona,
                    chat_history=st.session_state.chat_history,
                    user_message=user_input,
                    context=context
                )

                # Update histories
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

                # Keep memory window to last 20 messages (10 turns)
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]

            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error: {error_msg}")
                if "401" in error_msg or "invalid_api_key" in error_msg:
                    st.info("💡 Your API key looks incorrect. Check it at platform.openai.com/api-keys")
                elif "429" in error_msg:
                    st.info("💡 Rate limit or billing issue. Check your usage at platform.openai.com/usage")
                elif "model" in error_msg.lower():
                    st.info("💡 Try switching to gpt-4o-mini in the sidebar.")

        st.rerun()
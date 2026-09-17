"""
app.py
------
Streamlit chat app: "Daraz Customer Support Operations Assistant"

- Loads a PRE-BUILT FAISS index + metadata.json (produced by ingest.py).
  This app NEVER re-embeds or re-processes the source PDFs.
- Sidebar lets the user restrict retrieval to one or more knowledge-base
  sections (returns, delivery, refunds, sellers, payments, customer_support).
- Uses Groq's hosted "openai/gpt-oss-120b" model to generate answers
  grounded in the retrieved chunks.
- The Groq API key is read from Streamlit secrets (st.secrets) only.
  It is never rendered in an input box or echoed back to the user.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
INDEX_DIR = Path("faiss_index")
FAISS_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # must match the model used in ingest.py
GROQ_MODEL = "openai/gpt-oss-120b"

ALL_DEPARTMENTS = [
    "returns",
    "delivery",
    "refunds",
    "sellers",
    "payments",
    "customer_support",
]

TOP_K_FINAL = 5          # chunks actually sent to the LLM
OVER_FETCH_K = 50        # how many candidates to pull from FAISS before filtering by department

DARAZ_ORANGE = "#F85606"
DARAZ_DARK = "#1A1A1A"


# --------------------------------------------------------------------------
# Page setup + branding
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Daraz Support Assistant",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FAFAFA;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {DARAZ_DARK};
        }}
        section[data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}
        .daraz-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 20px;
            background: linear-gradient(90deg, {DARAZ_ORANGE} 0%, #FF8A3D 100%);
            border-radius: 10px;
            margin-bottom: 18px;
        }}
        .daraz-header h1 {{
            color: white;
            font-size: 24px;
            margin: 0;
        }}
        .daraz-header p {{
            color: #FFF3EB;
            margin: 0;
            font-size: 13px;
        }}
        .source-chip {{
            display: inline-block;
            background-color: #FFE9DC;
            color: {DARAZ_ORANGE};
            border: 1px solid {DARAZ_ORANGE};
            border-radius: 14px;
            padding: 2px 10px;
            margin: 2px 4px 2px 0;
            font-size: 12px;
            font-weight: 600;
        }}
        div.stButton > button {{
            background-color: {DARAZ_ORANGE};
            color: white;
            border: none;
        }}
        div.stButton > button:hover {{
            background-color: #D9490A;
            color: white;
        }}
    </style>
    <div class="daraz-header">
        <div style="font-size:32px;">🛍️</div>
        <div>
            <h1>Daraz Support Assistant</h1>
            <p>Internal operations assistant — answers grounded in the Daraz knowledge base</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resource loaders (run once per session/process)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_index_and_metadata():
    if not FAISS_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    index = faiss.read_index(str(FAISS_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


embedder = load_embedding_model()
faiss_index, metadata = load_index_and_metadata()
groq_client = get_groq_client()


# --------------------------------------------------------------------------
# Sidebar — knowledge base section filter
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Knowledge Base Sections")
    st.caption("Restrict search to specific sections, or leave all checked to search everything.")

    if "selected_departments" not in st.session_state:
        st.session_state.selected_departments = set(ALL_DEPARTMENTS)

    col_a, col_b = st.columns(2)
    if col_a.button("Select all", use_container_width=True):
        st.session_state.selected_departments = set(ALL_DEPARTMENTS)
    if col_b.button("Clear all", use_container_width=True):
        st.session_state.selected_departments = set()

    selected = set()
    for dept in ALL_DEPARTMENTS:
        checked = st.checkbox(
            dept.replace("_", " ").title(),
            value=dept in st.session_state.selected_departments,
            key=f"chk_{dept}",
        )
        if checked:
            selected.add(dept)
    st.session_state.selected_departments = selected

    st.divider()
    if faiss_index is not None:
        st.success(f"Index loaded — {faiss_index.ntotal} chunks")
    else:
        st.error("faiss_index/ not found. Run ingest.py first.")

    if groq_client is None:
        st.error("GROQ_API_KEY not found in secrets.")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve_chunks(query: str, departments: set, top_k: int = TOP_K_FINAL):
    if faiss_index is None or metadata is None or not departments:
        return []

    query_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(OVER_FETCH_K, faiss_index.ntotal)
    scores, indices = faiss_index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk_meta = metadata[idx]
        if chunk_meta["department"] not in departments:
            continue
        results.append({**chunk_meta, "score": float(score)})
        if len(results) >= top_k:
            break

    return results


# --------------------------------------------------------------------------
# LLM answer generation (Groq)
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Daraz Customer Support Operations Assistant.
You help Daraz support agents and operations staff quickly find accurate
answers about policies for returns, delivery, refunds, sellers, payments,
and customer support.

Rules:
- Answer ONLY using the information in the provided context chunks.
- If the context does not contain the answer, clearly say you couldn't
  find it in the knowledge base, and suggest which section the agent
  should check manually or escalate to.
- Be concise, structured, and practical — agents are reading this while
  on a live call or chat with a customer or seller.
- When relevant, mention which section/policy the answer comes from.
- Never invent policy details, numbers, or timelines that are not in
  the context.
"""


def build_context_block(chunks):
    parts = []
    for c in chunks:
        parts.append(
            f"[Section: {c['department']} | Source: {c['source_file']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list, history: list):
    if groq_client is None:
        return "⚠️ Groq API key is not configured. Please add GROQ_API_KEY to Streamlit secrets."

    if not chunks:
        return (
            "I couldn't find anything relevant in the selected knowledge base "
            "section(s) for that question. Try selecting more sections in the "
            "sidebar, or rephrase the question."
        )

    context_block = build_context_block(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # include a short window of prior turns for conversational context
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append(
        {
            "role": "user",
            "content": (
                f"Context from the Daraz knowledge base:\n\n{context_block}\n\n"
                f"Agent's question: {query}"
            ),
        }
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# Chat UI
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🛍️"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">{s["department"]} · {s["source_file"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)

query = st.chat_input("Ask about returns, delivery, refunds, sellers, payments, or support policy...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🛍️"):
        with st.spinner("Searching knowledge base..."):
            chunks = retrieve_chunks(query, st.session_state.selected_departments)
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, chunks, st.session_state.messages)

        st.markdown(answer)
        if chunks:
            chips = "".join(
                f'<span class="source-chip">{c["department"]} · {c["source_file"]}</span>'
                for c in chunks
            )
            st.markdown(chips, unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": chunks}
    )

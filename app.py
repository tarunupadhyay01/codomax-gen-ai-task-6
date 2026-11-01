"""CareerPilot - Generative AI Capstone (Streamlit UI).

LLM API + Prompt Engineering + RAG + Tools + Agent in one app.
Run with:  streamlit run app.py
"""
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

from agent import EXAMPLES, MODES, MODEL, Tools, run_agent
from rag import MAX_CHUNKS, build_chunks, get_client, index_documents, load_document

load_dotenv()
st.set_page_config(page_title="CareerPilot", page_icon="🧭", layout="wide")
st.title("🧭 CareerPilot: AI Career & Study Assistant")
st.caption("Gemini LLM + RAG + Tools + Agent | Codomax Digital Solutions - GenAI Capstone")


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:  # noqa: BLE001
        return None


try:
    client = get_client(get_api_key())
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# ---------- Session state: memory, documents and database for this user ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:10]
if "tools" not in st.session_state:
    db_path = os.path.join(tempfile.gettempdir(), f"careerpilot_{st.session_state.session_id}.db")
    st.session_state.tools = Tools(client, db_path)
st.session_state.setdefault("store", None)
st.session_state.setdefault("stats", None)
st.session_state.setdefault("contents", [])   # full conversation sent to the model
st.session_state.setdefault("messages", [])   # what is shown on screen

# ---------- Sidebar ----------
with st.sidebar:
    st.header("1. Your documents")
    files = st.file_uploader("Resume, job description or notes (PDF, TXT, MD)",
                             type=["pdf", "txt", "md"], accept_multiple_files=True)
    with st.expander("Chunking settings"):
        chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, 50)
        overlap = st.slider("Chunk overlap", 0, 400, 150, 10)

    if st.button("Process documents", type="primary", disabled=not files):
        try:
            with st.spinner("Reading, chunking and embedding..."):
                pages = []
                for f in files:
                    pages.extend(load_document(f.name, f.getvalue()))
                chunks = build_chunks(pages, chunk_size, overlap)
                if not chunks:
                    raise ValueError("No readable text found. Scanned PDFs need OCR.")
                if len(chunks) > MAX_CHUNKS:
                    st.warning(f"Large input: only the first {MAX_CHUNKS} chunks are used.")
                    chunks = chunks[:MAX_CHUNKS]
                st.session_state.store = index_documents(client, chunks)
                st.session_state.stats = {"files": len(files), "chunks": len(chunks)}
            st.success("Documents are ready.")
        except (RuntimeError, ValueError) as e:
            st.error(str(e))
    if st.session_state.stats:
        s = st.session_state.stats
        st.write(f"Files: **{s['files']}** | Chunks: **{s['chunks']}**")

    st.header("2. Choose a mode")
    mode = st.radio("Assistant mode", list(MODES), label_visibility="collapsed")
    st.caption(MODES[mode].split(". ", 1)[1])

    st.header("3. Try these")
    for ex in EXAMPLES[mode]:
        if st.button(ex, key=f"{mode}-{ex}"):
            st.session_state.pending = ex

    st.divider()
    st.subheader("My tasks")
    rows = st.session_state.tools.task_rows("all")
    if rows:
        for tid, title, due, done in rows:
            st.markdown(f"{'✅' if done else '⬜'} **{tid}.** {title}" + (f" _(due {due})_" if due else ""))
    else:
        st.caption("No tasks yet. Ask the agent to make a plan.")

    if st.button("Clear chat and tasks"):
        st.session_state.contents = []
        st.session_state.messages = []
        st.session_state.pop("tools", None)
        st.session_state.pop("pending", None)
        st.rerun()

# ---------- Show history ----------
if not st.session_state.messages:
    st.info("Upload your resume and a job description in the sidebar, click **Process documents**, "
            "then pick a mode and start chatting. Sample files are included in the project.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("steps"):
            with st.expander(f"Agent steps ({len(m['steps'])} tool call(s))"):
                for i, s in enumerate(m["steps"], start=1):
                    st.markdown(f"**{i}. Tool:** `{s['tool']}`  \n**Input:** `{s['args']}`")
                    st.code(s["result"], language=None)

# ---------- New message ----------
user_text = st.chat_input("Ask CareerPilot...") or st.session_state.pop("pending", None)
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("Thinking and using tools..."):
            try:
                tools = st.session_state.tools
                tools.store = st.session_state.store      # give the RAG tool the latest vector DB
                answer, steps, contents = run_agent(
                    client, tools, st.session_state.contents, user_text, mode, MODEL)
                st.session_state.contents = contents
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "steps": steps})
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))
                st.session_state.messages.pop()

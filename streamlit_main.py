import streamlit as st
import os
import base64
import re
import html
from main_rag import initialize_retriever, LLMClient, RAGPipeline

st.set_page_config(layout="wide")
st.title("📚 Agentic RAG - Technical Document Assistant")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Embedding Model",
    ["all-MiniLM-L6-v2"]
)

alpha_value = st.sidebar.slider(
    "Hybrid Alpha (Semantic vs Keyword Weight)",
    0.0, 1.0, 0.5, 0.1
)

top_k = st.sidebar.slider(
    "Top K Documents",
    1, 10, 5
)

# -----------------------------
# Retriever
# -----------------------------
@st.cache_resource
def load_retriever(model_name, alpha):
    os.environ["EMBED_MODEL"] = model_name
    os.environ["ALPHA"] = str(alpha)
    return initialize_retriever()

retriever = load_retriever(model_choice, alpha_value)

# -----------------------------
# Session State
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = None

if "final_answer" not in st.session_state:
    st.session_state.final_answer = None

if "selected_citation_idx" not in st.session_state:
    st.session_state.selected_citation_idx = 0

if "citation_selected" not in st.session_state:
    st.session_state.citation_selected = False


# -----------------------------
# Helpers
# -----------------------------
def extract_citations(answer_text: str):
    """Return ordered unique citation numbers found in answer text."""
    seen = []
    for match in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', answer_text):
        for n in match.group(1).split(","):
            n = int(n.strip())
            if n not in seen:
                seen.append(n)
    return seen


def render_answer_html(answer_text: str) -> str:
    """Render answer with [N] highlighted as styled spans."""
    def highlight_multi(match):
        inner = match.group(1)
        nums = [n.strip() for n in inner.split(",")]
        return "".join(
            f'<span style="color:#1a73e8;font-weight:700;background:#e8f0fe;'
            f'border-radius:4px;padding:1px 6px;margin:0 1px;">[{n}]</span>'
            for n in nums if n.isdigit()
        )

    def highlight_single(match):
        n = match.group(1)
        return (
            f'<span style="color:#1a73e8;font-weight:700;background:#e8f0fe;'
            f'border-radius:4px;padding:1px 6px;margin:0 1px;">[{n}]</span>'
        )

    # Escape the raw answer text FIRST before injecting any HTML
    safe = html.escape(answer_text)
    safe = re.sub(r'\[(\d+(?:,\s*\d+)+)\]', highlight_multi, safe)
    safe = re.sub(r'\[(\d+)\]', highlight_single, safe)
    safe = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', safe)
    safe = safe.replace('\n', '<br>')

    return f"""
    <div style="
        font-size:15px;line-height:1.9;color:#1a1a1a;
        background:#f9f9f9;border-left:4px solid #1a73e8;
        padding:16px 20px;border-radius:6px;margin-top:8px;
    ">{safe}</div>
    """


def render_chunk_card(i: int, r, is_active: bool):
    """Render a single retrieved chunk card safely."""
    border = "2px solid #1a73e8" if is_active else "1px solid #ddd"
    bg     = "#f0f6ff"           if is_active else "#ffffff"
    badge  = (
        '<span style="font-size:11px;color:#fff;background:#1a73e8;'
        'border-radius:3px;padding:1px 7px;margin-left:8px;">▶ viewing</span>'
        if is_active else ""
    )

    # Safely escape ALL user-sourced strings
    doc_name   = html.escape(str(r.doc_name))
    page_num   = html.escape(str(r.page_number))
    # Truncate then escape content — this is the key fix
    snippet    = html.escape(r.content[:450]) + "…"

    st.markdown(
        f"""
        <div style="border:{border};background:{bg};border-radius:8px;
                    padding:12px 16px;margin-bottom:12px;">
            <div style="display:flex;align-items:center;margin-bottom:4px;">
                <strong>[{i}] 📄 {doc_name} &nbsp;·&nbsp; Page {page_num}</strong>
                {badge}
            </div>
            <div style="font-size:12px;color:#666;margin-bottom:8px;">
                Hybrid:&nbsp;<code>{round(r.hybrid_score, 4)}</code>
                &nbsp;|&nbsp;
                Semantic:&nbsp;<code>{round(r.semantic_score, 4)}</code>
                &nbsp;|&nbsp;
                Keyword:&nbsp;<code>{round(r.keyword_score, 4)}</code>
            </div>
            <div style="font-size:14px;color:#333;line-height:1.6;
                        white-space:pre-wrap;word-break:break-word;">
                {snippet}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1, 1])

# =============================
# LEFT PANEL
# =============================
with left_col:
    st.header("💬 Ask a Question")

    user_query = st.text_input("Enter your query:")

    if st.button("Search"):
        if user_query.strip() == "":
            st.warning("Please enter a query.")
        else:
            with st.spinner("Retrieving documents and generating answer..."):
                llm = LLMClient()
                rag = RAGPipeline(retriever, llm)
                response = rag.run(user_query, top_k=top_k)
                st.session_state.results      = response["chunks"]
                st.session_state.final_answer = response["answer"]
                # Auto-open viewer on the first cited document if citations exist
                cited = extract_citations(response["answer"])
                valid = [n for n in cited if 1 <= n <= len(response["chunks"])]
                if valid:
                    st.session_state.selected_citation_idx = valid[0] - 1
                    st.session_state.citation_selected = True
                else:
                    st.session_state.selected_citation_idx = 0
                    st.session_state.citation_selected = False
            st.rerun()

    # --- Final Answer ---
    if st.session_state.final_answer:
        st.markdown("### ✅ Final Answer")

        st.markdown(
            render_answer_html(st.session_state.final_answer),
            unsafe_allow_html=True,
        )

        # One button per retrieved chunk — only shown when answer has citations
        cited_nums = extract_citations(st.session_state.final_answer)
        valid_cited = [n for n in cited_nums if 1 <= n <= len(st.session_state.results)]

        if st.session_state.results and valid_cited:
            st.markdown(
                '<p style="margin:10px 0 4px;font-size:13px;color:#555;">'
                '📎 <strong>Jump to source:</strong></p>',
                unsafe_allow_html=True,
            )
            btn_cols = st.columns(len(st.session_state.results))
            for col, (idx, chunk) in zip(btn_cols, enumerate(st.session_state.results)):
                n = idx + 1
                is_active = (idx == st.session_state.selected_citation_idx and st.session_state.citation_selected)
                label = f"[{n}] p.{chunk.page_number}"
                # Use type="primary" to visually highlight the active button
                btn_type = "primary" if is_active else "secondary"
                if col.button(label, key=f"jump_{n}", type=btn_type):
                    st.session_state.selected_citation_idx = idx
                    st.session_state.citation_selected = True
                    st.rerun()

    # --- Retrieved context chunks ---
    # if st.session_state.results:
    #     st.subheader("🧠 Retrieved Context")
    #     for i, r in enumerate(st.session_state.results, start=1):
    #         is_active = (i - 1 == st.session_state.selected_citation_idx)
    #         render_chunk_card(i, r, is_active)


# =============================
# RIGHT PANEL
# =============================
with right_col:
    st.header("📑 Document Viewer")

    if st.session_state.results and st.session_state.citation_selected:

        clamped_idx = min(
            st.session_state.selected_citation_idx,
            len(st.session_state.results) - 1
        )

        selected_doc = st.selectbox(
            "Select Document",
            st.session_state.results,
            index=clamped_idx,
            format_func=lambda x: f"{x.doc_name} | Page {x.page_number}"
        )

        local_path = os.path.join("RPD-en-US", selected_doc.doc_name)

        if os.path.exists(local_path):
            st.markdown("### 📄 Full PDF Document")

            with open(local_path, "rb") as f:
                pdf_bytes = f.read()

            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

            pdf_display = f"""
                <iframe
                    src="data:application/pdf;base64,{base64_pdf}#page={selected_doc.page_number}"
                    width="100%"
                    height="800px"
                    type="application/pdf">
                </iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)

        else:
            st.error(f"File not found in RPD-en-US: {selected_doc.doc_name}")

        st.markdown("### 📌 Metadata")
        st.json({
            "Document":       selected_doc.doc_name,
            "Page":           selected_doc.page_number,
            "Chunk ID":       selected_doc.chunk_id,
            "Hybrid Score":   selected_doc.hybrid_score,
            "Semantic Score": selected_doc.semantic_score,
            "Keyword Score":  selected_doc.keyword_score,
        })

    else:
        st.info("Click on the citation in the answer button above to view the source document here.")
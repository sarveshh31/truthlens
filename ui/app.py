import html
import os
import re

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
COLORS = {"supported": "#d9f2e3", "partial": "#fff0c2", "unsupported": "#fcdada"}
LABELS = {"supported": "supported", "partial": "partly supported", "unsupported": "NOT supported"}


def clean(text: str) -> str:
    """Show citations as [1] instead of the fullwidth brackets some models emit."""
    return re.sub(r"【(\d+)】", r"[\1]", text)


def render_sentences(sentences):
    for s in sentences:
        color = COLORS.get(s["verdict"], "#eee")
        label = LABELS.get(s["verdict"], s["verdict"])
        reason = html.escape(str(s.get("reason", "")))
        st.markdown(
            f"<div style='background:{color};padding:8px 12px;border-radius:6px;margin-bottom:6px;color:#111'>"
            f"{html.escape(clean(s['text']))}<br><small><b>{label}</b> - {reason}</small></div>",
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="TruthLens", layout="wide")
st.title("TruthLens")
st.caption("A RAG agent that fact-checks its own answer, sentence by sentence, against the document.")

with st.sidebar:
    st.header("Settings")
    filename = st.text_input("Document filename", "capgemini_report.pdf")
    k = st.slider("Passages to retrieve", 3, 8, 5)
    plant = st.checkbox("Demo: plant a false sentence", help="Injects a fake sentence into the first draft to show the verifier catching it.")
    st.markdown("**Legend**")
    for verdict, color in COLORS.items():
        st.markdown(f"<span style='background:{color};padding:2px 8px;border-radius:4px;color:#111'>{LABELS[verdict]}</span>", unsafe_allow_html=True)

examples = ["Who is the CEO?", "Which five technology trends does the Group highlight for 2026?",
            "What operating margin did the Group achieve in 2025?", "What is the recipe for chocolate cake?"]
choice = st.selectbox("Try an example or type your own below", [""] + examples)
question = st.text_input("Your question", value=choice)

if st.button("Ask", type="primary") and question.strip():
    payload = {"question": question.strip(), "k": k, "filename": filename or None, "debug_plant_lie": plant}
    with st.spinner("Retrieving, answering and fact-checking..."):
        try:
            r = requests.post(f"{API_URL}/ask", json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Could not get an answer from the API at {API_URL}: {e}")
            st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Trust score (final)", f"{data['trust']:.0%}")
    c2.metric("Trust before fixing", f"{data.get('first_draft_trust', data['trust']):.0%}")
    c3.metric("Rewrites used", data.get("rewrites", 0))

    first = data.get("first_sentences") or []
    changed = data.get("rewrites", 0) > 0 and first
    if changed:
        left, right = st.columns(2)
        with left:
            st.subheader("First draft (before fixing)")
            render_sentences(first)
        with right:
            st.subheader("Final answer (after fixing)")
            if data.get("refused"):
                st.warning(data["answer"])
            else:
                render_sentences(data["sentences"])
    else:
        st.subheader("Answer")
        if data.get("refused"):
            st.warning(data["answer"])
        elif data.get("sentences"):
            render_sentences(data["sentences"])
        else:
            st.write(clean(data["answer"]))

    with st.expander(f"Sources ({len(data.get('sources', []))} passages)"):
        for i, src in enumerate(data.get("sources", []), 1):
            tag = " (added: same page)" if src.get("expanded") else ""
            st.markdown(f"**[{i}] page {src['page']}**{tag}")
            st.write(src["content"][:700] + ("..." if len(src["content"]) > 700 else ""))
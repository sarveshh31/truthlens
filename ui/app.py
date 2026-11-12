import html
import os
import re

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

VERDICT = {
    "supported": {"bg": "rgba(214,245,228,.85)", "bar": "#2ea36b", "label": "Supported"},
    "partial": {"bg": "rgba(255,243,200,.9)", "bar": "#e0a600", "label": "Partly supported"},
    "unsupported": {"bg": "rgba(255,222,222,.9)", "bar": "#e5484d", "label": "Not supported"},
}
EXAMPLES = {
    "Who is the CEO?": "Who is the CEO?",
    "Five tech trends for 2026": "Which five technology trends does the Group highlight for 2026?",
    "Operating margin 2025": "What operating margin did the Group achieve in 2025?",
    "Recipe for chocolate cake": "What is the recipe for chocolate cake?",
}

st.set_page_config(page_title="TruthLens", page_icon="🔍", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;800;900&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea { font-family: 'Nunito', 'Segoe UI', sans-serif !important; }

.stApp {
  background:
    radial-gradient(circle at 4% 96%, rgba(70, 224, 186, .60), transparent 30%),
    radial-gradient(circle at 98% 12%, rgba(255, 84, 120, .55), transparent 28%),
    radial-gradient(circle at 96% 58%, rgba(255, 168, 108, .45), transparent 26%),
    radial-gradient(circle at 28% 18%, rgba(255, 228, 130, .38), transparent 22%),
    #ffffff;
  background-attachment: fixed;
}
.stApp::before, .stApp::after {
  content: ""; position: fixed; border: 1.5px solid rgba(0,0,0,.28); border-radius: 50%;
  pointer-events: none; z-index: 0;
}
.stApp::before { width: 520px; height: 520px; left: -330px; top: 38%; }
.stApp::after  { width: 520px; height: 520px; right: -330px; top: 22%; }

header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
.block-container { max-width: 980px !important; padding-top: 2.2rem !important; position: relative; z-index: 1; }

/* top bar */
.topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem; }
.dots { width: 42px; height: 42px;
  background-image: radial-gradient(circle, #555 1.6px, transparent 1.8px); background-size: 9px 9px; opacity: .8; }
.nav { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; justify-content: center; }
.chip { padding: .38rem 1rem; border-radius: 999px; font-size: .92rem; color: #444; }
.chip.on { border: 1.8px solid #111; color: #111; font-weight: 600; background: rgba(255,255,255,.7); }
.badge { font-weight: 800; letter-spacing: .5px; font-size: 1.05rem; color: #1f8f55; }

/* hero */
.hero { text-align: center; margin: 1.4rem 0 .4rem; }
.hero h1 { font-size: clamp(3.2rem, 11vw, 7.2rem); font-weight: 900; letter-spacing: -.04em; line-height: .95; margin: 0; color: #0b0b0b; }
.hero h2 { font-size: clamp(1.4rem, 3.6vw, 2.3rem); font-weight: 400; margin: .35rem 0 .6rem; color: #111; }
.hero p { max-width: 640px; margin: 0 auto; color: #777; font-size: .98rem; line-height: 1.6; }

/* inputs */
div[data-baseweb="input"] { border-radius: 999px !important; border: 1.8px solid #111 !important; background: rgba(255,255,255,.85) !important; }
div[data-baseweb="input"] > div { background: transparent !important; }
div[data-baseweb="input"] input { padding: .85rem 1.3rem !important; font-size: 1.05rem !important; }
.stButton > button { border-radius: 999px !important; border: 1.8px solid #111 !important; background: rgba(255,255,255,.85) !important;
  color: #111 !important; padding: .55rem 1.4rem !important; font-weight: 600 !important; transition: all .18s ease; }
.stButton > button:hover { background: #111 !important; color: #fff !important; transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: #111 !important; color: #fff !important; }
.stButton > button[kind="primary"]:hover { background: #333 !important; }
div[data-testid="stExpander"] { border: 1.5px solid rgba(0,0,0,.12) !important; border-radius: 18px !important; background: rgba(255,255,255,.7); }

/* metric cards */
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 1.4rem 0 1rem; }
.card { border-radius: 22px; padding: 1rem 1.3rem; background: rgba(255,255,255,.72); backdrop-filter: blur(8px);
  border: 1.5px solid rgba(0,0,0,.10); box-shadow: 0 6px 24px rgba(0,0,0,.05); }
.card .k { font-size: .82rem; color: #777; letter-spacing: .03em; text-transform: uppercase; }
.card .v { font-size: 2.3rem; font-weight: 800; line-height: 1.15; }

/* sentences */
.sec { font-size: 1.5rem; font-weight: 800; margin: 1.3rem 0 .6rem; }
.sent { border-radius: 16px; padding: .85rem 1.1rem; margin-bottom: 9px; color: #151515; line-height: 1.55;
  border: 1px solid rgba(0,0,0,.06); box-shadow: 0 4px 16px rgba(0,0,0,.04); }
.sent .why { display: block; margin-top: .35rem; font-size: .82rem; color: #444; }
.sent .tag { font-weight: 800; }
.legend { display: flex; gap: .5rem; flex-wrap: wrap; margin: .2rem 0 .4rem; }
.lg { font-size: .8rem; padding: .2rem .7rem; border-radius: 999px; color: #151515; border: 1px solid rgba(0,0,0,.08); }
.refuse { border-radius: 16px; padding: 1rem 1.2rem; background: rgba(255,243,200,.9); border-left: 6px solid #e0a600; color: #151515; }
.src { font-size: .88rem; color: #333; }
@media (max-width: 700px) { .cards { grid-template-columns: 1fr; } }
</style>
""",
    unsafe_allow_html=True,
)


def clean(text: str) -> str:
    """Show citations as [1] instead of the fullwidth brackets some models emit."""
    return re.sub(r"【(\d+)】", r"[\1]", text)


def sentence_cards(sentences):
    out = []
    for s in sentences:
        v = VERDICT.get(s["verdict"], VERDICT["partial"])
        why = html.escape(str(s.get("reason", "")))
        out.append(
            f"<div class='sent' style='background:{v['bg']};border-left:6px solid {v['bar']}'>"
            f"{html.escape(clean(s['text']))}<span class='why'><span class='tag'>{v['label']}</span> &middot; {why}</span></div>"
        )
    return "".join(out)


def trust_color(t: float) -> str:
    return "#2ea36b" if t >= 0.99 else ("#e0a600" if t >= 0.6 else "#e5484d")


def metric_cards(final, first, rewrites):
    return (
        "<div class='cards'>"
        f"<div class='card'><div class='k'>Trust score</div><div class='v' style='color:{trust_color(final)}'>{final:.0%}</div></div>"
        f"<div class='card'><div class='k'>Before fixing</div><div class='v' style='color:{trust_color(first)}'>{first:.0%}</div></div>"
        f"<div class='card'><div class='k'>Rewrites</div><div class='v'>{rewrites}</div></div>"
        "</div>"
    )


# ---------- top bar + hero ----------
st.markdown(
    """
<div class="topbar">
  <div class="dots"></div>
  <div class="nav"><span class="chip">RAG</span><span class="chip">LangGraph</span><span class="chip on">TruthLens</span>
  <span class="chip">pgvector</span><span class="chip">Verifier</span></div>
  <div class="badge">TL</div>
</div>
<div class="hero">
  <h1>TRUTHLENS</h1>
  <h2>answers that check themselves</h2>
  <p>Ask a question about the document. TruthLens writes an answer, fact-checks every sentence against the source,
  and fixes anything it cannot back up before showing it to you.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

if "q" not in st.session_state:
    st.session_state.q = ""


def pick(text):
    st.session_state.q = text
    st.session_state.go = True


cols = st.columns(len(EXAMPLES))
for i, (label, full) in enumerate(EXAMPLES.items()):
    cols[i].button(label, key=f"ex{i}", on_click=pick, args=(full,), use_container_width=True)

question = st.text_input("Your question", key="q", placeholder="Ask anything about the document...", label_visibility="collapsed")

left, mid, right = st.columns([1, 1, 1])
ask = mid.button("Ask  →", type="primary", use_container_width=True)

with st.expander("Settings"):
    s1, s2, s3 = st.columns([2, 1, 2])
    filename = s1.text_input("Document filename", "capgemini_report.pdf")
    k = s2.slider("Passages", 3, 8, 5)
    plant = s3.checkbox("Demo: plant a false sentence", help="Injects a fake sentence into the first draft to show the verifier catching it.")

st.markdown(
    "<div class='legend'>"
    + "".join(f"<span class='lg' style='background:{v['bg']}'>{v['label']}</span>" for v in VERDICT.values())
    + "</div>",
    unsafe_allow_html=True,
)

go = ask or st.session_state.pop("go", False)
if go and question.strip():
    payload = {"question": question.strip(), "k": k, "filename": filename or None, "debug_plant_lie": plant}
    with st.spinner("Retrieving, answering and fact-checking..."):
        try:
            r = requests.post(f"{API_URL}/ask", json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Could not get an answer from the API at {API_URL}: {e}")
            st.stop()

    first_trust = data.get("first_draft_trust", data["trust"])
    st.markdown(metric_cards(data["trust"], first_trust, data.get("rewrites", 0)), unsafe_allow_html=True)

    first = data.get("first_sentences") or []
    if data.get("rewrites", 0) > 0 and first:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='sec'>First draft</div>" + sentence_cards(first), unsafe_allow_html=True)
        with c2:
            body = (f"<div class='refuse'>{html.escape(data['answer'])}</div>" if data.get("refused")
                    else sentence_cards(data["sentences"]))
            st.markdown("<div class='sec'>After fixing</div>" + body, unsafe_allow_html=True)
    else:
        if data.get("refused"):
            body = f"<div class='refuse'>{html.escape(data['answer'])}</div>"
        elif data.get("sentences"):
            body = sentence_cards(data["sentences"])
        else:
            body = f"<div class='sent'>{html.escape(clean(data['answer']))}</div>"
        st.markdown("<div class='sec'>Answer</div>" + body, unsafe_allow_html=True)

    with st.expander(f"Sources ({len(data.get('sources', []))} passages)"):
        for i, src in enumerate(data.get("sources", []), 1):
            tag = " (added: same page)" if src.get("expanded") else ""
            text = html.escape(src["content"][:700]) + ("..." if len(src["content"]) > 700 else "")
            st.markdown(f"<div class='src'><b>[{i}] page {src['page']}</b>{tag}<br>{text}</div><br>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;color:#999;font-size:.8rem;margin-top:2.5rem'>TruthLens &middot; Sarvesh Tiwari</p>", unsafe_allow_html=True)
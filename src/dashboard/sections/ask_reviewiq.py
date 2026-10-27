"""Ask ReviewIQ: product-scoped question answering. The answer is always
shown with the actual review excerpts it was built from, and weak evidence
produces a "not enough information" reply instead of a guess."""
import html

import streamlit as st

from widgets import product_select

SUGGESTIONS = [
    "What are the biggest complaints?",
    "Is it good value for the price?",
    "How is the battery life?",
    "Does it have a 3 year warranty?",
]


@st.cache_resource(show_spinner="Loading the local chatbot model (first time only, ~1-2 min) ...")
def _load_chatbot():
    from rag_pipeline import RAGChatbot
    return RAGChatbot()


def _use_suggestion(question: str):
    st.session_state["ask_q"] = question
    st.session_state["ask_pending"] = True


def _ask_clicked():
    st.session_state["ask_pending"] = True


def render(table):
    parent_asin = product_select(table, "Ask about", key="ask")

    st.markdown('<div class="sec-title" style="margin-top:14px">Try a question</div>', unsafe_allow_html=True)
    for i, (col, q) in enumerate(zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS)):
        with col:
            with st.container(key=f"chip_{i}"):
                st.button(q, key=f"sugg_{i}", on_click=_use_suggestion, args=(q,), use_container_width=True)

    question = st.text_input("Your question", key="ask_q", placeholder="Ask about customer experiences...")
    st.button("Ask ReviewIQ", on_click=_ask_clicked)

    if st.session_state.pop("ask_pending", False) and question:
        bot = _load_chatbot()
        with st.spinner("Retrieving reviews and generating an answer ..."):
            result = bot.ask(question, parent_asin=parent_asin)

        st.markdown(
            f'<div class="answer-card"><div class="atag">REVIEWIQ ANSWER</div>{html.escape(result["answer"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sec-title">Evidence used</div>', unsafe_allow_html=True)
        for i, (_, row) in enumerate(result["supporting_reviews"].iterrows(), 1):
            st.markdown(f"**Review {i}** · similarity {row['similarity']:.2f} · rating {row['rating']}/5")
            st.markdown(f'<div class="evidence-card">{html.escape(str(row["clean_text"])[:300])}</div>',
                        unsafe_allow_html=True)

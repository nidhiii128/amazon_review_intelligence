"""Page 5: Ask ReviewIQ -- the RAG page. Product-scoped question -> answer
+ the actual retrieved evidence, so the answer is never an unverifiable
guess (and product isolation is enforced by scoping on parent_asin)."""
import streamlit as st

from data_helpers import display_name
from widgets import product_picker


@st.cache_resource(show_spinner="Loading the local chatbot model (first time only, ~1-2 min) ...")
def _load_chatbot():
    from rag_pipeline import RAGChatbot
    return RAGChatbot()


def render(reviews, products):
    st.subheader("🤖 Ask ReviewIQ")

    parent_asin = product_picker(reviews, products, "Selected Product", key="ask_product")
    st.caption(f"Asking about: **{display_name(products, parent_asin)}**")

    question = st.text_input(
        "Ask about customer experiences...",
        placeholder="What are the biggest complaints?",
    )
    if st.button("Ask ReviewIQ") and question:
        bot = _load_chatbot()
        with st.spinner("Retrieving reviews and generating an answer ..."):
            result = bot.ask(question, parent_asin=parent_asin)

        st.markdown("#### 🤖 REVIEWIQ ANSWER")
        st.success(result["answer"])

        st.markdown("---")
        st.markdown("#### 📚 Evidence Used")
        for i, (_, row) in enumerate(result["supporting_reviews"].iterrows(), 1):
            st.markdown(f"**Review {i}** — Similarity: {row['similarity']:.2f} "
                        f"(rating {row['rating']}/5)")
            st.markdown(
                f'<div class="evidence-card">{str(row["clean_text"])[:300]}</div>',
                unsafe_allow_html=True,
            )

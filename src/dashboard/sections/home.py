"""Home: the shop front. Search results, product shelves, and aspect tiles
("shop by what matters") that lead into the per-product insights."""
import re

import streamlit as st

from data_helpers import get_best_for_aspect
from widgets import card_row

# aspect -> (icon, gradient)
TILES = {
    "battery": ("🔋", "linear-gradient(135deg,#f5a300,#ef6c00)"),
    "sound": ("🔊", "linear-gradient(135deg,#7c3aed,#4f46e5)"),
    "camera": ("📷", "linear-gradient(135deg,#0f766e,#22c55e)"),
    "display": ("🖥️", "linear-gradient(135deg,#2563eb,#0ea5e9)"),
    "build_quality": ("🛠️", "linear-gradient(135deg,#475569,#1e293b)"),
}


def _pick_aspect(aspect: str):
    st.session_state["shelf_aspect"] = None if st.session_state.get("shelf_aspect") == aspect else aspect


def render(reviews, absa, table):
    hero = (
        f"""<div class="shop-hero"><div><h2>Find a product. See what reviewers really say.</h2>
        <p>Every product below is summarised from its real customer reviews.</p></div>
        <div class="chips"><span class="chip"><b>{len(reviews):,}</b> reviews</span>
        <span class="chip"><b>{len(table):,}</b> products</span>
        <span class="chip">Electronics</span></div></div>"""
    )
    st.markdown(re.sub(r"\n\s*", "", hero), unsafe_allow_html=True)

    query = (st.session_state.get("q") or "").strip()
    if query:
        hits = table[table["product_name"].str.contains(query, case=False, na=False, regex=False)]
        hits = hits.sort_values("review_count", ascending=False)
        st.markdown(f'<div class="sec-title">Results for “{query}” <small>{len(hits):,} products</small></div>',
                    unsafe_allow_html=True)
        if hits.empty:
            st.info("No products match that search. Try a shorter word such as “echo” or “speaker”.")
            return
        for i in range(0, min(len(hits), 15), 5):
            card_row(hits.iloc[i:i + 5], f"s{i}")
            st.write("")
        return

    st.markdown('<div class="sec-title">Most reviewed products</div>', unsafe_allow_html=True)
    card_row(table.sort_values("review_count", ascending=False), "most")

    st.markdown('<div class="sec-title">Highest rated <small>products with 15+ reviews</small></div>',
                unsafe_allow_html=True)
    top_rated = table[table["review_count"] >= 15].sort_values(
        ["avg_rating", "positive_pct", "review_count"], ascending=False)
    card_row(top_rated, "rated")

    st.markdown('<div class="sec-title">Shop by what matters</div>', unsafe_allow_html=True)
    selected = st.session_state.get("shelf_aspect")
    for col, (aspect, (icon, gradient)) in zip(st.columns(len(TILES)), TILES.items()):
        with col:
            with st.container(key=f"tile_{aspect}"):
                label = aspect.replace("_", " ").title()
                st.markdown(
                    f'<div class="tile{" sel" if selected == aspect else ""}" style="background:{gradient}">'
                    f'<div class="ticon">{icon}</div><div><small>BEST FOR</small><b>{label}</b></div></div>',
                    unsafe_allow_html=True,
                )
                st.button("pick", key=f"pick_{aspect}", on_click=_pick_aspect, args=(aspect,))

    if selected:
        best = get_best_for_aspect(absa, table, selected)
        label = selected.replace("_", " ").title()
        st.markdown(f'<div class="sec-title">Best for {label} <small>highest share of positive {label.lower()} comments</small></div>',
                    unsafe_allow_html=True)
        if best.empty:
            st.info("Not enough mentions of this aspect yet.")
        else:
            card_row(best, f"best_{selected}")

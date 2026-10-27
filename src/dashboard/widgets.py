"""Shared widgets for the shop-style dashboard: navigation constants, the
clickable product card / aspect tile, and the shared "current product"."""
import html

import streamlit as st

from theme import star_string, image_style

PAGE_HOME = "🏠 Shop"
PAGE_INSIGHTS = "📊 Product Insights"
PAGE_COMPARE = "⚖️ Compare"
PAGE_ASK = "💬 Ask ReviewIQ"
PAGE_SELLER = "📈 Seller Insights"
PAGES = [PAGE_HOME, PAGE_INSIGHTS, PAGE_COMPARE, PAGE_ASK, PAGE_SELLER]


def go_to(page: str, **state):
    """on_click callback: switch the sidebar page (and set any extra session
    state, e.g. the product to open) before the next run builds the widgets.
    The sidebar radio owns the "nav" key, so its identity never changes and
    a click is never dropped."""
    st.session_state["nav"] = page
    for key, value in state.items():
        st.session_state[key] = value


def open_product(parent_asin: str, page: str = PAGE_INSIGHTS):
    go_to(page, current_asin=parent_asin)


def product_card_html(row) -> str:
    name = html.escape(str(row.product_name))
    pos = int(row.positive_pct) if row.positive_pct == row.positive_pct else 0
    return (
        f'<div class="pc"><div class="pc-img" style="{image_style(getattr(row, "image_url", None))}"></div>'
        f'<div class="pc-name">{name}</div>'
        f'<div class="pc-meta"><span><span class="star">★</span> <b>{row.avg_rating}</b></span>'
        f'<span>{row.review_count:,} reviews</span></div>'
        f'<div class="pc-bar" style="--p:{pos}%"></div><div class="pc-pos">{pos}% positive</div></div>'
    )


def card_row(frame, shelf: str, n: int = 5):
    """A row of clickable product cards; clicking one opens its insights."""
    rows = list(frame.head(n).itertuples())
    cols = st.columns(n)
    for col, row in zip(cols, rows):
        with col:
            with st.container(key=f"pc_{shelf}_{row.parent_asin}"):
                st.markdown(product_card_html(row), unsafe_allow_html=True)
                st.button("open", key=f"open_{shelf}_{row.parent_asin}", on_click=open_product, args=(row.parent_asin,))


def product_select(table, label: str, key: str, default_asin: str = None, shared: bool = True, limit: int = 400):
    """A searchable product dropdown. When `shared`, it reads and writes the
    app-wide current product, so Insights, Compare and Ask stay on the same
    product as the user moves between pages. The widget state is re-seeded
    from the current product before the widget is built (never via index=,
    which would change the widget's identity)."""
    import pandas as pd
    top = table.sort_values("review_count", ascending=False).head(limit)
    current = st.session_state.get("current_asin") if shared else None
    if current in set(table["parent_asin"]) and current not in set(top["parent_asin"]):
        top = pd.concat([table[table["parent_asin"] == current], top])
    lookup = top.set_index("parent_asin")
    options = lookup.index.tolist()
    current = current if current in lookup.index else (default_asin or options[0])
    wkey = f"sel_{key}"
    if wkey not in st.session_state or (shared and st.session_state[wkey] != current):
        st.session_state[wkey] = current

    def _sync():
        if shared:
            st.session_state["current_asin"] = st.session_state[wkey]

    chosen = st.selectbox(
        label, options, key=wkey, on_change=_sync,
        format_func=lambda pa: f"{lookup.loc[pa, 'product_name'][:80]}   ·   {lookup.loc[pa, 'avg_rating']}★   ·   {lookup.loc[pa, 'review_count']} reviews",
    )
    if shared:
        st.session_state["current_asin"] = chosen
    return chosen

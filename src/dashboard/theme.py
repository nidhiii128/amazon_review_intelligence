"""
Shared styling + small render helpers used by every page under
src/dashboard/pages/. Kept separate from app.py so pages can import it
without a circular import back into the router.
"""
import pathlib

import pandas as pd
import streamlit as st

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures"
METRICS_DIR = BASE_DIR / "reports" / "metrics"

# Semantic status colors (validated for contrast/CVD-safety) -- used
# consistently everywhere positive/neutral/negative appears, in this app
# and in the matplotlib charts from earlier milestones.
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#fab219"
COLOR_CRITICAL = "#d03b3b"
COLOR_ACCENT = "#eb6834"
COLOR_ACCENT_DARK = "#d95926"
COLOR_NAVY = "#131a2b"
COLOR_NAVY_LIGHT = "#1f2a44"

STATUS_COLOR_MAP = [COLOR_GOOD, COLOR_WARNING, COLOR_CRITICAL]  # positive, neutral, negative
STATUS_EMOJI = {"positive": "🟢", "negative": "🔴", "mixed": "🟡"}

CUSTOM_CSS = f"""
<style>
.stApp {{ background-color: #f9f9f7; }}
div.block-container {{ padding-top: 1rem; max-width: 1200px; }}

.topbar {{
    background: linear-gradient(180deg, {COLOR_NAVY} 0%, {COLOR_NAVY_LIGHT} 100%);
    margin: -1rem -1rem 1.2rem -1rem;
    padding: 20px 32px 16px 32px;
    border-radius: 0 0 14px 14px;
    display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap;
}}
.topbar .brand {{ color: #ffffff; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }}
.topbar .brand span {{ color: {COLOR_ACCENT}; }}
.topbar .tagline {{ color: #b8c4d9; font-size: 13px; margin-top: 2px; }}
.topbar .stats {{ color: #e7ecf3; font-size: 13px; text-align: right; line-height: 1.5; }}
.topbar .stats b {{ color: {COLOR_ACCENT}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    background-color: white; border-radius: 8px 8px 0 0; padding: 10px 24px;
    font-weight: 600; font-size: 15px; border: 1px solid #e4e7eb; border-bottom: none;
}}
.stTabs [aria-selected="true"] {{
    background-color: {COLOR_ACCENT} !important; color: white !important;
}}

div.stButton > button {{
    background-color: {COLOR_ACCENT}; color: white; border: none;
    border-radius: 6px; font-weight: 700; padding: 6px 18px;
}}
div.stButton > button:hover {{ background-color: {COLOR_ACCENT_DARK}; color: white; }}

[data-testid="stMetric"] {{
    background: white; border-radius: 10px; padding: 12px 10px;
    border: 1px solid #e4e7eb;
}}

.product-card {{
    background: white; border: 1px solid #e4e7eb; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
}}
.product-card .pname {{ font-weight: 700; font-size: 14px; color: {COLOR_NAVY}; min-height: 36px; }}
.product-card .pstars {{ color: {COLOR_ACCENT}; font-size: 14px; margin-top: 2px; }}
.product-card .pmeta {{ color: #6b7280; font-size: 12px; margin-top: 2px; }}

.evidence-card {{
    background: white; border: 1px solid #e4e7eb; border-left: 4px solid {COLOR_ACCENT};
    border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px;
}}
.suspicious-card {{
    background: #fff8ec; border: 1px solid #f0d9a8; border-left: 4px solid {COLOR_WARNING};
    border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px;
}}
</style>
"""

PLACEHOLDER_IMAGE = (
    "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"
)


def star_string(rating: float) -> str:
    if rating is None or pd.isna(rating):
        return ""
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def show_image_if_exists(path: pathlib.Path, caption: str = None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not generated yet: run the script that produces `{path.name}`.")


def show_text_if_exists(path: pathlib.Path, expander_title: str):
    with st.expander(expander_title):
        if path.exists():
            st.text(path.read_text())
        else:
            st.info(f"Not generated yet: `{path.name}` doesn't exist. Run the corresponding script first.")


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_topbar(reviews: pd.DataFrame):
    st.markdown(
        f"""<div class="topbar">
            <div>
                <div class="brand">🧠 Review<span>IQ</span></div>
                <div class="tagline">AI-Based Product Review Intelligence and Consumer Insight System</div>
            </div>
            <div class="stats">
                <b>{len(reviews):,}</b> reviews analyzed &nbsp;|&nbsp;
                <b>{reviews['parent_asin'].nunique():,}</b> products &nbsp;|&nbsp;
                Electronics category sample
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

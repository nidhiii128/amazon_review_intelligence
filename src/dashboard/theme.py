"""
Shared styling + small render helpers used by every page under
src/dashboard/sections/. Kept separate from app.py so pages can import it
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
COLOR_ACCENT = "#f5a300"
COLOR_ACCENT_DARK = "#c98600"
COLOR_NAVY = "#131a2b"
COLOR_NAVY_LIGHT = "#1f2a44"

STATUS_COLOR_MAP = [COLOR_GOOD, COLOR_WARNING, COLOR_CRITICAL]  # positive, neutral, negative
STATUS_EMOJI = {"positive": "🟢", "negative": "🔴", "mixed": "🟡"}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stMarkdown, p, span, div, h1, h2, h3, h4, h5, h6,
label, button, input, textarea, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}
/* Never override Streamlit's own icon font (icons are font ligatures). */
[data-testid="stIconMaterial"], [data-testid^="stIcon"] {{
    font-family: 'Material Symbols Rounded' !important;
}}

.stApp {{ background: #f3f4f6; color: {COLOR_NAVY}; }}
/* thin accent strip across the very top, like a shop front */
.stApp::before {{
    content: ""; position: fixed; top: 0; left: 0; right: 0; height: 5px;
    background: {COLOR_ACCENT}; z-index: 1000000;
}}
[data-testid="stHeader"] {{ background: transparent; }}
div.block-container {{ padding-top: 2.6rem; max-width: 1280px; }}
.stApp h3, .stApp h4 {{ color: {COLOR_NAVY}; font-weight: 700; letter-spacing: -0.2px; }}

/* ---------- Sidebar (white, shop-style) ---------- */
section[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid #eceef2; }}
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] > div {{ padding-top: 0.6rem; }}
.sb-brand {{ text-align: center; font-size: 31px; font-weight: 800; letter-spacing: -1.2px; color: {COLOR_NAVY}; padding-top: 10px; }}
.sb-brand span {{ color: {COLOR_ACCENT}; }}
.sb-swoosh {{ display: block; margin: -4px auto 14px auto; width: 96px; height: 12px;
    background: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='96' height='12' viewBox='0 0 96 12'%3E%3Cpath d='M6 3 Q48 15 90 3' fill='none' stroke='%23f5a300' stroke-width='3.5' stroke-linecap='round'/%3E%3C/svg%3E") no-repeat center; }}
.sb-label {{ font-size: 11px; letter-spacing: 0.6px; text-transform: uppercase; color: #8a92a3; font-weight: 600; margin: 18px 0 2px 2px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 8px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: #f3f4f6; border-radius: 10px; padding: 11px 16px; width: 100%; margin: 0;
    transition: background 0.15s ease;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div > div:not([data-testid="stMarkdownContainer"]) {{ display: none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ color: {COLOR_NAVY}; font-size: 14px; font-weight: 600; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: #ffeec2; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: #fff4d6; box-shadow: inset 4px 0 0 {COLOR_ACCENT};
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ font-weight: 800; }}
section[data-testid="stSidebar"] [data-testid="stTextInput"] div[data-baseweb="input"] {{
    background: transparent; border: none; border-bottom: 2px solid #e6e8ec; border-radius: 0;
}}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input {{ font-size: 18px; font-weight: 700; color: {COLOR_NAVY}; padding-left: 2px; }}
.thumb {{ height: 56px; border-radius: 10px; background-color: #fafafa; background-repeat: no-repeat; background-position: center; background-size: contain; border: 1px solid #eef0f3; }}
.thumb-cap {{ font-size: 11px; color: {COLOR_NAVY}; font-weight: 700; text-align: center; margin-top: 4px; }}
.sb-stats {{ background: #f3f4f6; border-radius: 12px; padding: 12px 14px; margin-top: 16px; }}
.sb-stats .big {{ font-size: 22px; font-weight: 800; color: {COLOR_NAVY}; }}
.sb-stats .sub {{ font-size: 12px; color: #8a92a3; }}

/* ---------- Clickable cards: a transparent button laid over the markup ---------- */
[class*="st-key-pc_"], [class*="st-key-tile_"], [class*="st-key-thumb_"] {{ position: relative; }}
[class*="st-key-pc_"] [data-testid="stElementContainer"]:has(button),
[class*="st-key-tile_"] [data-testid="stElementContainer"]:has(button),
[class*="st-key-thumb_"] [data-testid="stElementContainer"]:has(button) {{
    position: absolute; inset: 0; z-index: 5; margin: 0;
}}
[class*="st-key-pc_"] button, [class*="st-key-tile_"] button, [class*="st-key-thumb_"] button {{
    width: 100%; height: 100%; opacity: 0; padding: 0; cursor: pointer;
}}
[class*="st-key-pc_"] [data-testid="stVerticalBlock"],
[class*="st-key-tile_"] [data-testid="stVerticalBlock"],
[class*="st-key-thumb_"] [data-testid="stVerticalBlock"] {{ gap: 0; }}
[class*="st-key-pc_"]:hover .pc, [class*="st-key-tile_"]:hover .tile {{ transform: translateY(-3px); box-shadow: 0 12px 26px rgba(20, 30, 50, 0.12); }}

/* ---------- Product cards ---------- */
.pc {{
    background: #fff; border-radius: 14px; padding: 16px 16px 14px 16px; height: 100%;
    box-shadow: 0 1px 2px rgba(20, 30, 50, 0.05); transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.pc-img {{ height: 128px; margin-bottom: 14px; background-color: #fff; background-repeat: no-repeat; background-position: center; background-size: contain; }}
.pc-name {{ font-size: 13px; font-weight: 600; color: {COLOR_NAVY}; line-height: 1.35; height: 36px; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }}
.pc-meta {{ display: flex; justify-content: space-between; align-items: baseline; margin-top: 8px; font-size: 12px; color: #8a92a3; }}
.pc-meta b {{ color: {COLOR_NAVY}; font-size: 13px; }}
.pc-meta .star {{ color: {COLOR_ACCENT}; }}
.pc-bar {{ height: 4px; border-radius: 3px; margin-top: 9px; background: linear-gradient(90deg, {COLOR_GOOD} var(--p), #e9ecf1 var(--p)); }}
.pc-pos {{ font-size: 11px; color: #8a92a3; margin-top: 4px; }}

/* ---------- Section titles and tiles ---------- */
.sec-title {{ font-size: 16px; font-weight: 700; color: {COLOR_NAVY}; margin: 28px 0 12px 2px; display: flex; align-items: baseline; gap: 10px; }}
.sec-title small {{ font-size: 12px; font-weight: 600; color: {COLOR_ACCENT_DARK}; }}
.tile {{
    height: 150px; border-radius: 14px; padding: 16px 18px; color: #fff; display: flex; flex-direction: column;
    justify-content: space-between; transition: transform 0.15s ease, box-shadow 0.15s ease; position: relative; overflow: hidden;
}}
.tile .ticon {{ font-size: 34px; }}
.tile small {{ font-size: 10px; letter-spacing: 1.6px; font-weight: 700; opacity: 0.85; display: block; }}
.tile b {{ font-size: 21px; font-weight: 800; }}
.tile.sel {{ outline: 3px solid {COLOR_NAVY}; outline-offset: 2px; }}

.shop-hero {{ background: #fff; border-radius: 16px; padding: 22px 26px; display: flex; justify-content: space-between; align-items: center; gap: 20px; flex-wrap: wrap; box-shadow: 0 1px 2px rgba(20,30,50,0.05); }}
.shop-hero h2 {{ margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.2px; word-spacing: 1px; color: {COLOR_NAVY}; }}
.shop-hero p {{ margin: 4px 0 0 0; color: #6b7280; font-size: 14px; }}
.chips {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.chip {{ background: #f3f4f6; border-radius: 999px; padding: 7px 14px; font-size: 13px; color: {COLOR_NAVY}; font-weight: 600; }}
.chip b {{ color: {COLOR_ACCENT_DARK}; }}

/* ---------- Product detail header ---------- */
.detail {{ background: #fff; border-radius: 16px; padding: 22px 26px; display: grid; grid-template-columns: 170px 1fr; gap: 26px; align-items: center; box-shadow: 0 1px 2px rgba(20,30,50,0.05); margin-bottom: 14px; }}
.detail .dimg {{ height: 150px; background-color: #fff; background-repeat: no-repeat; background-position: center; background-size: contain; }}
.detail h3 {{ margin: 0 0 6px 0; font-size: 20px; line-height: 1.3; }}
.detail .stars {{ color: {COLOR_ACCENT}; font-size: 17px; font-weight: 700; }}
@media (max-width: 800px) {{ .detail {{ grid-template-columns: 1fr; }} }}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #e3e6eb; }}
.stTabs [data-baseweb="tab"] {{ background: transparent; border: none; padding: 10px 18px; font-weight: 600; font-size: 15px; color: #6b7280; }}
.stTabs [data-baseweb="tab"]:hover {{ color: {COLOR_NAVY}; background: transparent; }}
.stTabs [aria-selected="true"] {{ color: {COLOR_NAVY} !important; background: transparent !important; box-shadow: inset 0 -3px 0 {COLOR_ACCENT}; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ---------- Buttons ---------- */
div.stButton > button {{
    background-color: {COLOR_ACCENT}; color: {COLOR_NAVY}; border: none;
    border-radius: 10px; font-weight: 700; padding: 8px 20px;
    transition: transform 0.12s ease, background 0.12s ease;
}}
div.stButton > button:hover {{ background-color: {COLOR_ACCENT_DARK}; color: {COLOR_NAVY}; transform: translateY(-1px); }}
div.stButton > button:active {{ transform: translateY(0); }}
div[class*="st-key-chip_"] div.stButton > button {{ background: #fff; border: 1px solid #e3e6eb; color: {COLOR_NAVY}; font-weight: 600; font-size: 13px; padding: 8px 14px; border-radius: 999px; min-height: 48px; }}
div[class*="st-key-chip_"] div.stButton > button:hover {{ background: #fff4d6; border-color: {COLOR_ACCENT}; color: {COLOR_NAVY}; transform: none; }}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {{ background: #fff; border-radius: 12px; padding: 14px 16px; border: 1px solid #eceef2; }}

/* ---------- Evidence / trust cards ---------- */
.evidence-card {{
    background: #fff; border: 1px solid #eceef2; border-left: 4px solid {COLOR_ACCENT};
    border-radius: 8px; padding: 12px 15px; margin-bottom: 8px; font-size: 13px;
}}
.answer-card {{ background: #fff; border-radius: 14px; padding: 18px 22px; border-left: 5px solid {COLOR_GOOD}; box-shadow: 0 1px 2px rgba(20,30,50,0.05); font-size: 16px; line-height: 1.5; }}
.answer-card .atag {{ font-size: 11px; letter-spacing: 1px; font-weight: 800; color: {COLOR_GOOD}; margin-bottom: 4px; }}
</style>
"""

# Inline SVG data URI -- no network dependency. (The previous Wikimedia URL
# returned 403 to hotlinked requests, so products without an image showed a
# broken-image icon.)
PLACEHOLDER_IMAGE = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300' viewBox='0 0 300 300'>"
    "<rect width='300' height='300' fill='%23f1efeb'/>"
    "<rect x='90' y='100' width='120' height='90' rx='10' fill='none' stroke='%23b8b4ab' stroke-width='6'/>"
    "<circle cx='125' cy='128' r='10' fill='%23b8b4ab'/>"
    "<path d='M98 182 l35 -34 l25 24 l18 -16 l26 26' fill='none' stroke='%23b8b4ab' stroke-width='6' stroke-linejoin='round'/>"
    "<text x='150' y='228' font-family='sans-serif' font-size='16' fill='%23a09b90' text-anchor='middle'>No image available</text>"
    "</svg>"
).replace("'", "%27").replace("<", "%3C").replace(">", "%3E")


def image_style(image_url) -> str:
    """CSS background layers: the real image on top of an inline placeholder,
    so a missing or broken URL still shows the placeholder (an <img onerror>
    fallback would be stripped by Streamlit's markdown sanitizer)."""
    layers = [f"url('{PLACEHOLDER_IMAGE}')"]
    if isinstance(image_url, str) and image_url:
        layers.insert(0, f"url('{image_url}')")
    return "background-image:" + ",".join(layers) + ";"


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


def render_sidebar_brand():
    st.sidebar.markdown(
        '<div class="sb-brand">Review<span>IQ</span></div><span class="sb-swoosh"></span>',
        unsafe_allow_html=True,
    )

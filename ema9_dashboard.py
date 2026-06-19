"""
EMA9 Screener — Professional Dashboard v2.0
============================================
Enhanced UI · Efficient Rendering · WhatsApp Export

Run:
    pip install streamlit pandas plotly requests yfinance gspread google-auth
    streamlit run ema9_dashboard.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, io, os, json, html as html_module
from datetime import datetime, date
import yfinance as yf

try:
    import gspread
    from google.oauth2.service_account import Credentials as SACredentials
    _GSPREAD_OK = True
except ImportError:
    _GSPREAD_OK = False

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="EMA9 Screener Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
#  ENHANCED CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main { 
    background: #070B14 !important; 
    font-family: 'Inter', sans-serif !important;
    color: #E2E8F0 !important;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1421; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #38BDF8; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1421 !important;
    border-right: 1px solid #1E2D45 !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

/* Remove default padding */
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* Hide streamlit chrome */
//#MainMenu, footer, header { visibility: hidden; }

/* ── HEADER ── */
.dash-header {
    background: linear-gradient(135deg, #0D1421 0%, #0F1E35 50%, #0D1421 100%);
    border: 1px solid #1E3A5F;
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.dash-header::after {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.dash-title { 
    font-size: 2rem; font-weight: 900; 
    background: linear-gradient(135deg, #38BDF8, #818CF8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.dash-subtitle { font-size: 0.78rem; color: #64748B; margin-top: 4px; }
.live-dot { 
    display:inline-block; width:8px; height:8px; background:#22C55E;
    border-radius:50%; margin-right:6px; 
    box-shadow: 0 0 8px rgba(34,197,94,0.6);
    animation: pulse 2s infinite; 
}
@keyframes pulse { 
    0%,100%{opacity:1; transform:scale(1)} 
    50%{opacity:.4; transform:scale(0.7)} 
}

/* ── KPI CARDS ── */
.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 24px; }
.kpi-card {
    background: rgba(13, 20, 33, 0.9);
    backdrop-filter: blur(10px);
    border: 1px solid #1E2D45;
    border-radius: 16px;
    padding: 22px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.kpi-card:hover { 
    border-color: #38BDF8; 
    transform: translateY(-3px); 
    box-shadow: 0 10px 30px rgba(56, 189, 248, 0.1);
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.blue::before  { background: linear-gradient(90deg, #38BDF8, #818CF8); }
.kpi-card.green::before { background: linear-gradient(90deg, #22C55E, #16A34A); }
.kpi-card.amber::before { background: linear-gradient(90deg, #F59E0B, #D97706); }
.kpi-card.rose::before  { background: linear-gradient(90deg, #F43F5E, #E11D48); }
.kpi-card.purple::before{ background: linear-gradient(90deg, #A78BFA, #7C3AED); }
.kpi-label { font-size: 0.7rem; font-weight: 600; color: #64748B; 
    text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 2rem; font-weight: 800; color: #F1F5F9; 
    line-height: 1.1; margin: 6px 0 4px; }
.kpi-delta { font-size: 0.72rem; font-weight: 500; }
.kpi-delta.pos { color: #22C55E; } 
.kpi-delta.neg { color: #F43F5E; }
.kpi-delta.neutral { color: #64748B; }
.kpi-icon { position: absolute; right: 16px; top: 16px; 
    font-size: 1.6rem; opacity: 0.12; }

/* ── SECTION HEADER ── */
.section-title {
    font-size: 0.8rem; font-weight: 700; color: #38BDF8;
    text-transform: uppercase; letter-spacing: 0.1em;
    display: flex; align-items: center; gap: 8px;
    margin: 28px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1E2D45;
}

/* ── TABLE ── */
.table-container {
    border-radius: 16px;
    border: 1px solid #1E2D45;
    overflow: auto;
    max-height: 600px;
}
.signal-table { width: 100%; border-collapse: separate; border-spacing: 0; }
.signal-table th {
    background: #0F1E35; color: #64748B; font-size: 0.68rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    padding: 12px 16px; text-align: left; 
    border-bottom: 2px solid #1E3A5F;
    position: sticky; top: 0; z-index: 10;
}
.signal-table td {
    padding: 10px 16px; font-size: 0.82rem; color: #CBD5E1;
    border-bottom: 1px solid #111827;
    white-space: nowrap;
}
.signal-table tbody tr:nth-child(even) td {
    background: rgba(15, 30, 53, 0.3);
}
.signal-table tbody tr:hover td { 
    background: rgba(56, 189, 248, 0.06); 
}
.badge-prime { background: rgba(34,197,94,0.15); color: #22C55E;
    padding: 3px 10px; border-radius: 20px; font-size: 0.62rem; font-weight: 700;
    border: 1px solid rgba(34,197,94,0.3); white-space: nowrap; }
.badge-other { background: rgba(56,189,248,0.1); color: #38BDF8;
    padding: 3px 10px; border-radius: 20px; font-size: 0.62rem; font-weight: 700;
    border: 1px solid rgba(56,189,248,0.2); white-space: nowrap; }
.badge-up   { background: rgba(34,197,94,0.1);  color: #22C55E;
    padding: 2px 8px; border-radius: 4px; font-size: 0.62rem; font-weight: 600; white-space: nowrap; }
.badge-side { background: rgba(245,158,11,0.1); color: #F59E0B;
    padding: 2px 8px; border-radius: 4px; font-size: 0.62rem; font-weight: 600; white-space: nowrap; }
.badge-down { background: rgba(244,63,94,0.1);  color: #F43F5E;
    padding: 2px 8px; border-radius: 4px; font-size: 0.62rem; font-weight: 600; white-space: nowrap; }
.gap-pos { color: #22C55E; font-weight: 700; }
.gap-neg { color: #F43F5E; font-weight: 700; }
.ticker-cell { font-weight: 800; color: #38BDF8; font-size: 0.85rem; }

/* ── WHATSAPP SECTION ── */
.wa-section {
    background: linear-gradient(135deg, rgba(13, 20, 33, 0.95), rgba(15, 30, 53, 0.95));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(37, 222, 102, 0.15);
    border-radius: 18px;
    padding: 28px;
    margin: 28px 0;
    position: relative;
    overflow: hidden;
}
.wa-section::before {
    content: '';
    position: absolute;
    top: -30%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(37, 222, 102, 0.05) 0%, transparent 70%);
    pointer-events: none;
}
.wa-title {
    font-size: 1.3rem; font-weight: 800;
    background: linear-gradient(135deg, #25D366, #128C7E);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: flex; align-items: center; gap: 10px;
}
.wa-subtitle { font-size: 0.8rem; color: #64748B; margin-top: 4px; }

/* Chart container */
.chart-wrap {
    background: rgba(13, 20, 33, 0.9);
    backdrop-filter: blur(10px);
    border: 1px solid #1E2D45;
    border-radius: 16px; padding: 4px; margin-bottom: 20px;
}

/* Streamlit overrides */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background: #0F1E35 !important;
    border: 1px solid #1E2D45 !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label { 
    color: #64748B !important; font-size: 0.72rem !important; 
    font-weight: 600 !important;
}
[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    border: 1px solid #1E2D45 !important;
    background: #0F1E35 !important;
    color: #CBD5E1 !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    border-color: #38BDF8 !important;
    color: #38BDF8 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #0D1421 !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid #1E2D45 !important;
    padding: 8px 24px !important;
    font-weight: 600 !important;
    color: #64748B !important;
}
.stTabs [aria-selected="true"] {
    background: #0F1E35 !important;
    border-bottom: 2px solid #38BDF8 !important;
    color: #38BDF8 !important;
}

/* Radio */
[data-testid="stRadio"] label { font-size: 0.78rem !important; color: #64748B !important; }

/* Code block */
[data-testid="stCodeBlock"] {
    border-radius: 12px !important;
    border: 1px solid #1E2D45 !important;
    background: #0D1421 !important;
}
[data-testid="stCodeBlock"] pre {
    font-size: 0.78rem !important;
}

/* Toggle */
[data-testid="stToggle"] label { font-size: 0.8rem !important; color: #CBD5E1 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
SHEET_ID   = "1zWYm5lEWz5LuiiItbvpekOV7T_wQNB7RCbEK9uScD24"
GID        = "1020278800"
SHEET_NAME = "EMA9 Signals"
CSV_URL    = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

CHART_BG  = "#070B14"
PAPER_BG  = "#0D1421"
GRID_CLR  = "#1E2D45"
FONT_CLR  = "#94A3B8"
GREEN     = "#22C55E"
RED       = "#F43F5E"
BLUE      = "#38BDF8"
AMBER     = "#F59E0B"
PURPLE    = "#A78BFA"

NUM_EMOJIS = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]

def num_emoji(i):
    if 1 <= i <= 10:
        return NUM_EMOJIS[i-1]
    return f"{i}."


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════
def _get_sa_creds():
    sa_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "service_account.json")
    if os.path.exists(sa_file):
        return SACredentials.from_service_account_file(
            sa_file, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    ev = os.environ.get("GOOGLE_SA_JSON")
    if ev:
        return SACredentials.from_service_account_info(
            json.loads(ev), scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    return None

@st.cache_data(ttl=120, show_spinner=False)
def load_data() -> pd.DataFrame:
    try:
        creds = _get_sa_creds() if _GSPREAD_OK else None
        if creds:
            gc  = gspread.authorize(creds)
            ws  = gc.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
            df  = pd.DataFrame(ws.get_all_records())
        else:
            resp = requests.get(CSV_URL, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        rmap = {"fv_gap_%": "fv_gap_pct", "candles_ago": "candles_ago"}
        df = df.rename(columns={k: v for k, v in rmap.items() if k in df.columns})

        df["scan_time"]   = pd.to_datetime(df.get("scan_time"), errors="coerce")
        df["scan_date"]   = df["scan_time"].dt.date
        df["price"]       = pd.to_numeric(df.get("price"),      errors="coerce")
        df["fair_value"]  = pd.to_numeric(df.get("fair_value"), errors="coerce")
        df["fv_gap_pct"]  = pd.to_numeric(df.get("fv_gap_pct"), errors="coerce")
        df["candles_ago"] = pd.to_numeric(df.get("candles_ago"),errors="coerce")
        df["upside_pct"]  = ((df["fair_value"] - df["price"]) / df["price"] * 100).round(2)
        df["ticker"]      = df.get("ticker", pd.Series()).astype(str).str.upper().str.strip()
        df["type"]        = df.get("type",       pd.Series()).fillna("OTHER").str.upper()
        df["trend"]       = df.get("trend",      pd.Series()).fillna("")
        df["valuation"]   = df.get("valuation",  pd.Series()).fillna("")

        return df.dropna(subset=["ticker"]).sort_values("scan_time", ascending=False).reset_index(drop=True)
    except Exception as ex:
        st.error(f"❌ Data load failed: {ex}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_candles(ticker: str) -> pd.DataFrame:
    for suffix in [".NS", ".BO"]:
        try:
            t    = yf.Ticker(f"{ticker}{suffix}")
            hist = t.history(period="6mo", auto_adjust=True)
            if hist is not None and len(hist) > 20:
                hist = hist[["Open","High","Low","Close","Volume"]].copy().dropna()
                hist.index = pd.to_datetime(hist.index)
                return hist
        except Exception:
            continue
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════
def _layout(fig, height=420, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=BLUE, family="Inter"), x=0.01),
        plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, family="Inter", size=11),
        height=height, margin=dict(l=12, r=12, t=40, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_CLR, font=dict(size=10)),
        xaxis=dict(showgrid=False, showline=False, color=FONT_CLR, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=FONT_CLR, zeroline=False),
    )
    return fig


def build_candlestick(ticker: str, df_info: pd.Series) -> go.Figure:
    hist = load_candles(ticker)
    if hist.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No price data for {ticker}", showarrow=False,
                           font=dict(color=FONT_CLR, size=14), xref="paper", yref="paper", x=0.5, y=0.5)
        return _layout(fig, 520)

    hist["ema9"]  = hist["Close"].ewm(span=9,  adjust=False).mean()
    hist["sma50"] = hist["Close"].rolling(50).mean()

    bo_date = None
    if df_info is not None and "breakout_date" in df_info:
        try:
            bo_date = pd.to_datetime(df_info["breakout_date"])
        except Exception:
            pass

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28],
                        vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"],  close=hist["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
        name="Price", line=dict(width=1),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["ema9"],
        line=dict(color="#F59E0B", width=1.8),
        name="EMA 9",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["sma50"],
        line=dict(color=PURPLE, width=1.4, dash="dot"),
        name="SMA 50",
    ), row=1, col=1)

    if df_info is not None:
        fv = df_info.get("fair_value")
        if fv and pd.notna(fv):
            fig.add_hline(y=float(fv), line_color=BLUE, line_dash="dash",
                          line_width=1.2, row=1, col=1,
                          annotation_text=f"FV ₹{float(fv):,.1f}",
                          annotation_font_color=BLUE,
                          annotation_bgcolor="rgba(56,189,248,0.1)")

    if bo_date is not None:
        mask = hist.index.normalize() == bo_date.normalize()
        if mask.any():
            bo_row = hist[mask].iloc[0]
            fig.add_trace(go.Scatter(
                x=[bo_row.name], y=[bo_row["Low"] * 0.985],
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=14, color=GREEN),
                text=["EMA9 BO"], textposition="bottom center",
                textfont=dict(color=GREEN, size=9),
                name="Breakout",
                showlegend=False,
            ), row=1, col=1)

    vol_colors = [GREEN if c >= o else RED
                  for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"],
        marker_color=vol_colors, marker_opacity=0.6,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, family="Inter", size=11),
        height=520, margin=dict(l=12, r=12, t=12, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=False, color=FONT_CLR, zeroline=False,
                     rangebreaks=[dict(bounds=["sat","mon"])])
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR, color=FONT_CLR, zeroline=False)
    fig.update_yaxes(row=2, col=1, tickformat=".2s")
    return fig


def build_fv_bar(df: pd.DataFrame) -> go.Figure:
    d = df[df["type"] == "PRIME"].copy()
    if d.empty:
        d = df.copy()
    d = d.dropna(subset=["fv_gap_pct"]).sort_values("fv_gap_pct", ascending=True).tail(18)
    colors = [GREEN if v >= 0 else RED for v in d["fv_gap_pct"]]
    fig = go.Figure(go.Bar(
        x=d["fv_gap_pct"], y=d["ticker"], orientation="h",
        marker_color=colors, marker_opacity=0.85,
        text=[f"{v:+.1f}%" for v in d["fv_gap_pct"]],
        textposition="outside", textfont=dict(size=10, color=FONT_CLR),
        hovertemplate="<b>%{y}</b><br>FV Gap: %{x:.1f}%<extra></extra>",
    ))
    fig = _layout(fig, 400, "🎯 FV Gap % — Prime Targets")
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.add_vline(x=0, line_color=GRID_CLR, line_width=1)
    return fig


def build_scatter(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["price","fair_value"]).copy()
    if d.empty:
        return go.Figure()
    color_map = {"PRIME": GREEN, "OTHER": BLUE}
    fig = go.Figure()
    for t, grp in d.groupby("type"):
        fig.add_trace(go.Scatter(
            x=grp["price"], y=grp["fair_value"],
            mode="markers",
            marker=dict(color=color_map.get(t, BLUE), size=9,
                        opacity=0.8, line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
            name=t,
            hovertemplate="<b>%{text}</b><br>Price: ₹%{x:,.1f}<br>FV: ₹%{y:,.1f}<extra></extra>",
            text=grp["ticker"],
        ))
    mn = min(d["price"].min(), d["fair_value"].min()) * 0.95
    mx = max(d["price"].max(), d["fair_value"].max()) * 1.05
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                             mode="lines", line=dict(color=GRID_CLR, dash="dot", width=1.5),
                             name="Fair Value Line", hoverinfo="skip"))
    fig = _layout(fig, 370, "💹 Price vs Fair Value")
    fig.update_layout(xaxis_title="Price (₹)", yaxis_title="Fair Value (₹)")
    return fig


def build_donut(labels, values, colors, title) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color=PAPER_BG, width=3)),
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value} signals (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, family="Inter"),
        title=dict(text=title, font=dict(size=12, color=BLUE), x=0.5, xanchor="center"),
        height=300, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.05,
                    xanchor="center", x=0.5, font=dict(size=10)),
        showlegend=True,
    )
    return fig


def build_daily_bar(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby(["scan_date","type"]).size().reset_index(name="count")
    fig = go.Figure()
    for t, clr in [("PRIME", GREEN), ("OTHER", BLUE)]:
        d = daily[daily["type"] == t]
        fig.add_trace(go.Bar(x=d["scan_date"], y=d["count"],
                             name=t, marker_color=clr, marker_opacity=0.8,
                             hovertemplate=f"<b>{t}</b><br>%{{x}}<br>%{{y}} signals<extra></extra>"))
    fig = _layout(fig, 300, "📅 Daily Signal Count")
    fig.update_layout(barmode="stack", xaxis_title=None, yaxis_title="Signals")
    return fig


# ═══════════════════════════════════════════════════════════════
#  WHATSAPP TEXT GENERATOR
# ═══════════════════════════════════════════════════════════════
def generate_whatsapp_text(df: pd.DataFrame, fmt: str = "detailed",
                           include_all: bool = True) -> str:
    """Generate WhatsApp-formatted trade list for easy copy-paste."""
    lines = []

    last_scan = df["scan_time"].max()
    date_str = last_scan.strftime("%d %b %Y") if pd.notna(last_scan) else "—"
    time_str = last_scan.strftime("%H:%M") if pd.notna(last_scan) else "—"

    prime_df = df[df["type"] == "PRIME"].copy().sort_values("fv_gap_pct", ascending=False)
    other_df = df[df["type"] != "PRIME"].copy().sort_values("fv_gap_pct", ascending=False)

    if fmt == "summary":
        lines.append("📊 *EMA9 Daily Summary*")
        lines.append(f"📅 {date_str} | 🕒 {time_str}")
        lines.append("━━━━━━━━━━━━━━━")
        lines.append(f"🎯 Prime: {len(prime_df)} | 📋 Total: {len(df)}")
        up_count = len(df[df["trend"] == "UPTREND"])
        side_count = len(df[df["trend"] == "SIDEWAYS"])
        down_count = len(df[df["trend"] == "DOWNTREND"])
        lines.append(f"📈 Uptrend: {up_count} | ↔️ Sideways: {side_count} | 📉 Down: {down_count}")
        avg_gap = prime_df["fv_gap_pct"].mean()
        if pd.notna(avg_gap):
            lines.append(f"💰 Avg FV Gap: {avg_gap:+.1f}%")
        if not prime_df.empty:
            best = prime_df.iloc[0]
            lines.append(f"🏆 Best: *{best['ticker']}* {best['fv_gap_pct']:+.1f}%")
        lines.append("━━━━━━━━━━━━━━━")
        lines.append("🤖 EMA9 Screener")

    elif fmt == "compact":
        lines.append(f"📊 EMA9 SIGNALS | {date_str}")
        lines.append(f"🔥 Primes: {len(prime_df)} | Total: {len(df)}")
        lines.append("")
        if not prime_df.empty:
            lines.append("🎯 *PRIME:*")
            for i, (_, r) in enumerate(prime_df.head(30).iterrows(), 1):
                p = f"₹{r['price']:,.0f}" if pd.notna(r.get("price")) else "—"
                fv = f"₹{r['fair_value']:,.0f}" if pd.notna(r.get("fair_value")) else "—"
                g = f"{r['fv_gap_pct']:+.1f}%" if pd.notna(r.get("fv_gap_pct")) else "—"
                lines.append(f"{i}. *{r['ticker']}* {p}→{fv} ({g})")
        if include_all and not other_df.empty:
            lines.append("")
            lines.append("📋 *OTHER:*")
            for i, (_, r) in enumerate(other_df.head(30).iterrows(), 1):
                p = f"₹{r['price']:,.0f}" if pd.notna(r.get("price")) else "—"
                fv = f"₹{r['fair_value']:,.0f}" if pd.notna(r.get("fair_value")) else "—"
                g = f"{r['fv_gap_pct']:+.1f}%" if pd.notna(r.get("fv_gap_pct")) else "—"
                lines.append(f"{i}. *{r['ticker']}* {p}→{fv} ({g})")
        lines.append("")
        lines.append("🤖 EMA9 Screener")

    else:  # detailed
        lines.append("📊 *EMA9 SCREENING REPORT*")
        lines.append("━━━━━━━━━━━━━━━")
        lines.append(f"📅 Date: {date_str}")
        lines.append(f"🕒 Scan: {time_str}")
        lines.append(f"🎯 Prime Targets: {len(prime_df)} | Total: {len(df)}")
        lines.append("")

        if not prime_df.empty:
            lines.append(f"🔥 *PRIME TARGETS* ({len(prime_df)})")
            lines.append("")
            for i, (_, r) in enumerate(prime_df.head(50).iterrows(), 1):
                p = f"₹{r['price']:,.2f}" if pd.notna(r.get("price")) else "—"
                fv = f"₹{r['fair_value']:,.2f}" if pd.notna(r.get("fair_value")) else "—"
                g = f"{r['fv_gap_pct']:+.1f}%" if pd.notna(r.get("fv_gap_pct")) else "—"
                u = f"{r['upside_pct']:+.1f}%" if pd.notna(r.get("upside_pct")) else "—"
                trend = str(r.get("trend", ""))
                val = str(r.get("valuation", ""))

                lines.append(f"{num_emoji(i)} *{r['ticker']}*")
                lines.append(f"   💰 Price: {p} → FV: {fv}")
                lines.append(f"   📈 FV Gap: {g} | Upside: {u}")
                if trend:
                    lines.append(f"   📊 Trend: {trend}")
                if val:
                    lines.append(f"   💎 Valuation: {val}")
                lines.append("")

        if include_all and not other_df.empty:
            lines.append(f"📋 *OTHER SIGNALS* ({len(other_df)})")
            lines.append("")
            for i, (_, r) in enumerate(other_df.head(30).iterrows(), 1):
                p = f"₹{r['price']:,.0f}" if pd.notna(r.get("price")) else "—"
                fv = f"₹{r['fair_value']:,.0f}" if pd.notna(r.get("fair_value")) else "—"
                g = f"{r['fv_gap_pct']:+.1f}%" if pd.notna(r.get("fv_gap_pct")) else "—"
                lines.append(f"{i}. *{r['ticker']}* — {p} → {fv} ({g})")
            lines.append("")

        lines.append("━━━━━━━━━━━━━━━")
        lines.append("🤖 Generated by EMA9 Screener")
        lines.append("⚡ chartink + EMA9 BO + FV Engine")

    return "\n".join(lines)


def generate_single_ticker_whatsapp(row: dict) -> str:
    """Generate WhatsApp text for a single ticker."""
    ticker = str(row.get("ticker", ""))
    price = row.get("price")
    fv = row.get("fair_value")
    gap = row.get("fv_gap_pct")
    upside = row.get("upside_pct")
    trend = str(row.get("trend", ""))
    val = str(row.get("valuation", ""))
    stime = row.get("scan_time")

    lines = []
    lines.append(f"📊 *{ticker}* — EMA9 Signal")
    lines.append("━━━━━━━━━━━━━━━")
    if pd.notna(price):
        lines.append(f"💰 Price: ₹{price:,.2f}")
    if pd.notna(fv):
        lines.append(f"🎯 Fair Value: ₹{fv:,.2f}")
    if pd.notna(gap):
        lines.append(f"📈 FV Gap: {gap:+.1f}%")
    if pd.notna(upside):
        lines.append(f"🚀 Upside: {upside:+.1f}%")
    if trend:
        lines.append(f"📊 Trend: {trend}")
    if val:
        lines.append(f"💎 Valuation: {val}")
    if pd.notna(stime):
        lines.append(f"🕒 Scan: {stime.strftime('%d %b %Y, %H:%M')}")
    lines.append("━━━━━━━━━━━━━━━")
    lines.append("🤖 EMA9 Screener")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  COPY BUTTON COMPONENT (JS-based clipboard)
# ═══════════════════════════════════════════════════════════════
_COPY_BTN_TEMPLATE = """
<div style="text-align:center; padding:8px 0;">
    <div id="copy-text-__KEY__" style="display:none;">__TEXT__</div>
    <button id="btn-__KEY__" onclick="copyToWA('__KEY__')" style="
        background: linear-gradient(135deg, #25D366, #128C7E);
        color: white; border: none; padding: 16px 48px;
        border-radius: 14px; font-size: 1.05rem; font-weight: 700;
        cursor: pointer; transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(37, 222, 102, 0.35);
        font-family: Inter, sans-serif; letter-spacing: 0.02em;
        display: inline-flex; align-items: center; gap: 8px;
    ">📋 Copy for WhatsApp</button>
</div>
<script>
function copyToWA(key) {
    var el = document.getElementById('copy-text-' + key);
    var text = el.textContent || el.innerText;
    var btn = document.getElementById('btn-' + key);

    function showSuccess() {
        btn.innerHTML = '✅ Copied! Paste in WhatsApp';
        btn.style.background = 'linear-gradient(135deg, #22C55E, #16A34A)';
        setTimeout(function() {
            btn.innerHTML = '📋 Copy for WhatsApp';
            btn.style.background = 'linear-gradient(135deg, #25D366, #128C7E)';
        }, 2500);
    }

    function fallbackCopy() {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        ta.style.top = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        try { document.execCommand('copy'); showSuccess(); }
        catch(e) {
            btn.innerHTML = '⚠️ Select text below & copy manually';
            setTimeout(function() {
                btn.innerHTML = '📋 Copy for WhatsApp';
            }, 3000);
        }
        document.body.removeChild(ta);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(showSuccess).catch(fallbackCopy);
    } else {
        fallbackCopy();
    }
}
</script>
"""

def render_copy_button(text: str, key: str = "wa_copy"):
    """Render a styled copy-to-clipboard button using JS."""
    escaped = html_module.escape(text)
    html_code = (_COPY_BTN_TEMPLATE
                 .replace("__KEY__", key)
                 .replace("__TEXT__", escaped))
    components.html(html_code, height=80)


# ═══════════════════════════════════════════════════════════════
#  HTML TABLE BUILDER (Efficient single-render approach)
# ═══════════════════════════════════════════════════════════════
def trend_badge(t):
    if t == "UPTREND":   return '<span class="badge-up">↑ UP</span>'
    if t == "SIDEWAYS":  return '<span class="badge-side">→ SIDE</span>'
    if t == "DOWNTREND": return '<span class="badge-down">↓ DOWN</span>'
    return f'<span style="color:#64748B">{t}</span>'

def type_badge(t):
    if t == "PRIME": return '<span class="badge-prime">★ PRIME</span>'
    return '<span class="badge-other">OTHER</span>'

def gap_fmt(v):
    if pd.isna(v): return "—"
    cls = "gap-pos" if v >= 0 else "gap-neg"
    arrow = "▲" if v >= 0 else "▼"
    return f'<span class="{cls}">{arrow} {v:+.1f}%</span>'


def build_html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    """Build a complete HTML table — far more efficient than per-row st.columns."""
    if df.empty:
        return '<div style="text-align:center;padding:40px;color:#64748B;font-size:0.9rem">' \
               'No signals match current filters.</div>'

    val_colors = {"UNDERVALUED": "#22C55E", "FAIR": "#F59E0B", "OVERVALUED": "#F43F5E"}

    rows_html = []
    for idx, row in df.head(max_rows).iterrows():
        ticker = str(row.get("ticker", ""))
        price  = row.get("price")
        fv     = row.get("fair_value")
        gap    = row.get("fv_gap_pct")
        upside = row.get("upside_pct")
        val    = str(row.get("valuation", ""))
        trend  = str(row.get("trend", ""))
        ago    = row.get("candles_ago")
        stime  = row.get("scan_time")
        stype  = str(row.get("type", ""))

        price_str  = f"₹{price:,.2f}" if pd.notna(price) else "—"
        fv_str     = f"₹{fv:,.2f}" if pd.notna(fv) else "—"
        gap_str    = gap_fmt(gap)
        upside_str = gap_fmt(upside)
        vc = val_colors.get(val.upper(), "#64748B")
        val_str = f'<span style="color:{vc};font-weight:600">{val}</span>' if val else "—"
        trend_html = trend_badge(trend)
        type_html = type_badge(stype)
        ago_str = str(int(ago)) if pd.notna(ago) else "—"
        ts_str = stime.strftime("%d %b %H:%M") if pd.notna(stime) else "—"

        rows_html.append(
            f'<tr>'
            f'<td>{type_html} <span class="ticker-cell">{html_module.escape(ticker)}</span></td>'
            f'<td>{price_str}</td>'
            f'<td>{fv_str}</td>'
            f'<td>{gap_str}</td>'
            f'<td>{upside_str}</td>'
            f'<td>{val_str}</td>'
            f'<td>{trend_html}</td>'
            f'<td style="color:#64748B">{ago_str}</td>'
            f'<td style="color:#475569;font-size:0.72rem">{ts_str}</td>'
            f'</tr>'
        )

    table_html = (
        '<div class="table-container">'
        '<table class="signal-table">'
        '<thead><tr>'
        '<th>Ticker</th><th>Price ₹</th><th>Fair Value ₹</th>'
        '<th>FV Gap %</th><th>Upside %</th><th>Valuation</th>'
        '<th>Trend</th><th>Ago</th><th>Scan Time</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>'
    )
    return table_html


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════
if "sel_ticker" not in st.session_state:
    st.session_state.sel_ticker = None
if "sel_row" not in st.session_state:
    st.session_state.sel_row = None

# ── Load data ─────────────────────────────────────────────────
with st.spinner("Loading signals..."):
    raw_df = load_data()

if raw_df.empty:
    st.error("❌ No data available. Please check your connection or Google Sheet.")
    st.stop()

# ── HEADER ───────────────────────────────────────────────────
last_scan = raw_df["scan_time"].max()
last_scan_str = last_scan.strftime("%d %b %Y, %H:%M") if pd.notna(last_scan) else "—"
total_primes = len(raw_df[raw_df["type"] == "PRIME"])

st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-title">📈 EMA9 Signal Dashboard</div>
    <div class="dash-subtitle">
      <span class="live-dot"></span>Live · Chartink + EMA9 Breakout + Fair Value · 
      Last scan: <b style="color:#CBD5E1">{last_scan_str}</b>
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:2.2rem;font-weight:900;color:{GREEN}">{total_primes}</div>
    <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em">Prime Targets Today</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── FILTER BAR ───────────────────────────────────────────────
fc1, fc2, fc3, fc4, fc5, fc6, fc7 = st.columns([2, 1.3, 1.2, 1.2, 1.2, 1.2, 1])

with fc1:
    search_q = st.text_input("🔍 Search", "", placeholder="e.g. RELIANCE")
with fc2:
    all_dates = ["All Dates"] + [str(d) for d in sorted(raw_df["scan_date"].dropna().unique(), reverse=True)]
    sel_date  = st.selectbox("📅 Date", all_dates)
with fc3:
    sel_type  = st.selectbox("🏷 Type", ["All","PRIME","OTHER"])
with fc4:
    trends    = ["All"] + sorted(raw_df["trend"].dropna().unique().tolist())
    sel_trend = st.selectbox("📊 Trend", trends)
with fc5:
    vals      = ["All"] + sorted(raw_df["valuation"].dropna().unique().tolist())
    sel_val   = st.selectbox("💰 Valuation", vals)
with fc6:
    sort_options = ["FV Gap ↓", "FV Gap ↑", "Upside ↓", "Ticker A-Z", "Latest Scan"]
    sort_sel = st.selectbox("↕ Sort", sort_options)
with fc7:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Apply filters ─────────────────────────────────────────────
df = raw_df.copy()
if search_q:
    df = df[df["ticker"].str.contains(search_q.upper(), na=False)]
if sel_date != "All Dates":
    df = df[df["scan_date"] == date.fromisoformat(sel_date)]
if sel_type != "All":
    df = df[df["type"] == sel_type]
if sel_trend != "All":
    df = df[df["trend"] == sel_trend]
if sel_val != "All":
    df = df[df["valuation"] == sel_val]

# Apply sort
if sort_sel == "FV Gap ↓":
    df = df.sort_values("fv_gap_pct", ascending=False)
elif sort_sel == "FV Gap ↑":
    df = df.sort_values("fv_gap_pct", ascending=True)
elif sort_sel == "Upside ↓":
    df = df.sort_values("upside_pct", ascending=False)
elif sort_sel == "Ticker A-Z":
    df = df.sort_values("ticker", ascending=True)
elif sort_sel == "Latest Scan":
    df = df.sort_values("scan_time", ascending=False)

prime_df = df[df["type"] == "PRIME"]

# ── KPI CARDS ────────────────────────────────────────────────
avg_gap  = prime_df["fv_gap_pct"].mean()
best_row = prime_df.loc[prime_df["fv_gap_pct"].idxmax()] if not prime_df.empty else None
best_txt = f"{best_row['ticker']} {best_row['fv_gap_pct']:+.1f}%" if best_row is not None else "—"
up_count = len(df[df["trend"] == "UPTREND"])

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="kpi-icon">📡</div>
    <div class="kpi-label">Total Signals</div>
    <div class="kpi-value">{len(df)}</div>
    <div class="kpi-delta neutral">{df['ticker'].nunique()} unique tickers</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-icon">🎯</div>
    <div class="kpi-label">Prime Targets</div>
    <div class="kpi-value">{len(prime_df)}</div>
    <div class="kpi-delta pos">{len(prime_df)/max(len(df),1)*100:.0f}% of filtered signals</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-icon">💰</div>
    <div class="kpi-label">Avg FV Gap (Prime)</div>
    <div class="kpi-value">{f'{avg_gap:+.1f}%' if pd.notna(avg_gap) else '—'}</div>
    <div class="kpi-delta {'pos' if pd.notna(avg_gap) and avg_gap>0 else 'neg'}">
      {'Undervalued on avg' if pd.notna(avg_gap) and avg_gap>0 else 'Overvalued on avg'}
    </div>
  </div>
  <div class="kpi-card rose">
    <div class="kpi-icon">🏆</div>
    <div class="kpi-label">Best Opportunity</div>
    <div class="kpi-value" style="font-size:1.3rem">{best_txt}</div>
    <div class="kpi-delta pos">Highest FV gap</div>
  </div>
  <div class="kpi-card purple">
    <div class="kpi-icon">📈</div>
    <div class="kpi-label">In Uptrend</div>
    <div class="kpi-value">{up_count}</div>
    <div class="kpi-delta neutral">{up_count/max(len(df),1)*100:.0f}% of signals</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── CHARTS ROW 1 ─────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Analytics</div>', unsafe_allow_html=True)

ch1, ch2, ch3 = st.columns([2.5, 1.4, 1.4])

with ch1:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(build_fv_bar(df), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with ch2:
    type_counts = df["type"].value_counts()
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(
        build_donut(type_counts.index.tolist(), type_counts.values.tolist(),
                    [GREEN if t=="PRIME" else BLUE for t in type_counts.index],
                    "Signal Mix"),
        use_container_width=True, config={"displayModeBar": False}
    )
    st.markdown('</div>', unsafe_allow_html=True)

with ch3:
    trend_counts = df["trend"].value_counts()
    tcolors = {t: c for t, c in [("UPTREND", GREEN), ("SIDEWAYS", AMBER), ("DOWNTREND", RED)]}
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(
        build_donut(trend_counts.index.tolist(), trend_counts.values.tolist(),
                    [tcolors.get(t, BLUE) for t in trend_counts.index],
                    "Trend Regime"),
        use_container_width=True, config={"displayModeBar": False}
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── CHARTS ROW 2 ─────────────────────────────────────────────
ch4, ch5 = st.columns([3, 2])
with ch4:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(build_scatter(df), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with ch5:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(build_daily_bar(df), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  📱 WHATSAPP EXPORT SECTION
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="wa-section">
  <div class="wa-title">📱 WhatsApp Export</div>
  <div class="wa-subtitle">Copy all trades and paste directly into WhatsApp — formatted with emojis & bold text</div>
</div>
""", unsafe_allow_html=True)

wa_col1, wa_col2, wa_col3 = st.columns([1.2, 1, 1])

with wa_col1:
    wa_format = st.radio(
        "📋 Message Format",
        ["Detailed", "Compact", "Summary"],
        horizontal=False,
        help="Detailed: full info per trade | Compact: one line per trade | Summary: stats only"
    )

with wa_col2:
    wa_include_all = st.toggle(
        "Include ALL signals", 
        value=True,
        help="If OFF, only PRIME targets are included"
    )
    wa_limit = st.slider("Max trades per category", 10, 50, 30, step=5,
                         help="Limit to avoid WhatsApp message size limits")

with wa_col3:
    # Generate the text
    wa_fmt_lower = wa_format.lower()
    wa_text = generate_whatsapp_text(df, fmt=wa_fmt_lower, include_all=wa_include_all)
    wa_chars = len(wa_text)
    wa_lines = wa_text.count("\n") + 1
    st.markdown(f"""
    <div style="background:#0F1E35;border:1px solid #1E2D45;border-radius:12px;padding:14px;margin-top:22px">
        <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Message Stats</div>
        <div style="font-size:0.85rem;color:#CBD5E1">
            📝 {wa_lines} lines · {wa_chars:,} characters<br>
            {'✅ Within WhatsApp limit' if wa_chars < 65000 else '⚠️ May exceed WhatsApp limit'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Generate text with limit applied
def generate_whatsapp_text_limited(df, fmt, include_all, limit):
    """Wrapper to apply trade limits."""
    # Temporarily limit the data
    prime_d = df[df["type"] == "PRIME"].sort_values("fv_gap_pct", ascending=False).head(limit)
    other_d = df[df["type"] != "PRIME"].sort_values("fv_gap_pct", ascending=False).head(limit) if include_all else pd.DataFrame()
    limited_df = pd.concat([prime_d, other_d], ignore_index=True) if not other_d.empty else prime_d
    return generate_whatsapp_text(limited_df, fmt=fmt, include_all=include_all)

wa_text_final = generate_whatsapp_text_limited(df, wa_fmt_lower, wa_include_all, wa_limit)

# Preview
st.markdown("#### 📋 Preview")
st.code(wa_text_final, language="text")

# Copy button
render_copy_button(wa_text_final, key="wa_main_copy")

# Additional actions
wa_btn1, wa_btn2, wa_btn3 = st.columns(3)
with wa_btn1:
    st.markdown(f"""
    <a href="https://web.whatsapp.com" target="_blank" style="
        display:block;text-align:center;background:#0F1E35;border:1px solid #1E2D45;
        color:#25D366;padding:12px;border-radius:10px;text-decoration:none;font-weight:600;
        font-size:0.85rem;transition:all 0.2s;">
        🌐 Open WhatsApp Web
    </a>
    """, unsafe_allow_html=True)
with wa_btn2:
    st.download_button(
        "💾 Download as .txt",
        data=wa_text_final,
        file_name=f"ema9_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )
with wa_btn3:
    st.download_button(
        "📊 Download CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f"ema9_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")


# ═══════════════════════════════════════════════════════════════
#  SIGNAL TABLE (Efficient HTML rendering)
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🎯 Signal Table</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs([f"★ Prime Targets ({len(prime_df)})", f"All Signals ({len(df)})"])

with tab1:
    st.markdown(build_html_table(prime_df), unsafe_allow_html=True)

with tab2:
    st.markdown(build_html_table(df), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CANDLESTICK CHART PANEL
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🕯️ Candlestick Chart</div>', unsafe_allow_html=True)

# Ticker selection
all_tickers = sorted(df["ticker"].unique().tolist())
if all_tickers:
    sel_col1, sel_col2 = st.columns([3, 1])
    with sel_col1:
        default_idx = 0
        if st.session_state.sel_ticker and st.session_state.sel_ticker in all_tickers:
            default_idx = all_tickers.index(st.session_state.sel_ticker)
        selected_ticker = st.selectbox("📈 Select ticker to view chart", all_tickers,
                                        index=default_idx, key="ticker_select")
        st.session_state.sel_ticker = selected_ticker
        # Find the row
        match_rows = df[df["ticker"] == selected_ticker]
        if not match_rows.empty:
            st.session_state.sel_row = match_rows.iloc[0].to_dict()
    with sel_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.session_state.sel_row:
            single_wa = generate_single_ticker_whatsapp(st.session_state.sel_row)
            if st.button("📋 Copy This Ticker", use_container_width=True,
                        help="Copy this ticker's info for WhatsApp"):
                render_copy_button(single_wa, key="wa_single_copy")
                st.success("✅ Ticker info copied! Check the button above.")

    # Display chart
    if st.session_state.sel_ticker and st.session_state.sel_row:
        tk  = st.session_state.sel_ticker
        row = st.session_state.sel_row

        info_row = pd.Series(row)
        price = row.get("price"); fv = row.get("fair_value"); gap = row.get("fv_gap_pct")
        trend = row.get("trend",""); val = row.get("valuation","")

        # Info cards
        i1, i2, i3, i4, i5 = st.columns(5)
        with i1:
            st.markdown(f"""<div class="kpi-card blue" style="padding:14px">
                <div class="kpi-label">Price</div>
                <div class="kpi-value" style="font-size:1.4rem">₹{price:,.2f}</div>
            </div>""" if pd.notna(price) else "", unsafe_allow_html=True)
        with i2:
            st.markdown(f"""<div class="kpi-card green" style="padding:14px">
                <div class="kpi-label">Fair Value</div>
                <div class="kpi-value" style="font-size:1.4rem">₹{fv:,.2f}</div>
            </div>""" if pd.notna(fv) else "", unsafe_allow_html=True)
        with i3:
            gc = "#22C55E" if pd.notna(gap) and gap>=0 else "#F43F5E"
            st.markdown(f"""<div class="kpi-card amber" style="padding:14px">
                <div class="kpi-label">FV Gap %</div>
                <div class="kpi-value" style="font-size:1.4rem;color:{gc}">{f'{gap:+.1f}%' if pd.notna(gap) else '—'}</div>
            </div>""" if pd.notna(gap) else "", unsafe_allow_html=True)
        with i4:
            tc = {"UPTREND":"#22C55E","SIDEWAYS":"#F59E0B","DOWNTREND":"#F43F5E"}.get(trend,"#64748B")
            st.markdown(f"""<div class="kpi-card purple" style="padding:14px">
                <div class="kpi-label">Trend</div>
                <div class="kpi-value" style="font-size:1.2rem;color:{tc}">{trend or '—'}</div>
            </div>""", unsafe_allow_html=True)
        with i5:
            vc = {"UNDERVALUED":"#22C55E","FAIR":"#F59E0B","OVERVALUED":"#F43F5E"}.get(val.upper(),"#64748B")
            st.markdown(f"""<div class="kpi-card rose" style="padding:14px">
                <div class="kpi-label">Valuation</div>
                <div class="kpi-value" style="font-size:1.1rem;color:{vc}">{val or '—'}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

        with st.spinner(f"Loading {tk} chart..."):
            chart = build_candlestick(tk, info_row)

        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(chart, use_container_width=True,
                        config={"displayModeBar": True, "displaylogo": False,
                                "modeBarButtonsToRemove": ["autoScale2d","lasso2d","select2d"]})
        st.markdown('</div>', unsafe_allow_html=True)

        st.caption(f"🟡 EMA9 · 🟣 SMA50 · 🔵 Fair Value Line · "
                   f"Green candle = bullish · Red candle = bearish · "
                   f"▲ = EMA9 Breakout point")

        # Single ticker WhatsApp copy
        st.markdown("---")
        st.markdown("##### 📱 Copy This Ticker for WhatsApp")
        render_copy_button(single_wa if st.session_state.sel_row else "", key="wa_single_static")
else:
    st.info("No tickers available with current filters.")

# ── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px;padding:16px;border-top:1px solid #1E2D45;
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
  <div style="font-size:0.72rem;color:#374151">
    EMA9 Screener Dashboard v2.0 · Chartink + yfinance + Fair Value Engine
  </div>
  <div style="font-size:0.72rem;color:#374151">Auto-refreshes every 2 min · WhatsApp Export Ready</div>
</div>
""", unsafe_allow_html=True)
"""Visual language of the Smart Traffic dashboard.

Dark-mode glassmorphic design with animated gradients, pulsing badges,
and a premium feel suitable for a final-year project demo.

Contains the colour palette, CSS, small HTML components (top bar, KPI
cards, section headings) and one Plotly styling helper, plus a shim that
keeps the app working on Streamlit versions before and after 1.46, where
`use_container_width=True` became `width="stretch"`.
"""
import streamlit as st

# ---------------------------------------------------------------- palette
# Dark-mode core
BG       = "#0a0f1e"      # page background (deep navy)
SURFACE  = "#111827"      # card background
SURFACE2 = "#1a2236"      # elevated card / input background
BORDER   = "#1e2d4a"      # subtle border
BORDER2  = "#2a3f6a"      # hover / active border

# Text
INK   = "#f0f4ff"         # headings and axis text
BODY  = "#c5cde8"         # body text
MUTED = "#6b7fa8"         # secondary text

# Accent palette (vibrant on dark)
ACCENT  = "#3b82f6"       # primary blue
INDIGO  = "#6366f1"       # indigo
TEAL    = "#14b8a6"       # teal
GREEN   = "#22c55e"       # success green
AMBER   = "#f59e0b"       # warning amber
RED     = "#ef4444"       # danger red
VIOLET  = "#a855f7"       # violet (MAPPO highlight)
ROSE    = "#f43f5e"       # rose (emergency)

# Legacy aliases kept for backward compatibility
PRIMARY = ACCENT
SLATE   = "#94a3b8"
LINE    = BORDER
GRID    = BORDER

CONTROLLER_COLORS = {
    "Fixed-time":                        "#64748b",
    "Actuated (sensor)":                 "#14b8a6",
    "MARL (no forecast)":                "#6366f1",
    "MARL + forecast (proposed)":        "#3b82f6",
    "MARL + forecast, no safety layer":  "#f59e0b",
    "MAPPO + forecast (proposed)":       "#a855f7",
}

# Road colour by number of halted vehicles
QUEUE_SCALE = [
    (0,   "#1e3a5f"),
    (2,   "#1d4ed8"),
    (5,   "#f59e0b"),
    (9,   "#f97316"),
    (10**9, "#ef4444"),
]


def queue_color(halted):
    for limit, colour in QUEUE_SCALE:
        if halted <= limit:
            return colour
    return QUEUE_SCALE[-1][1]


def heat_color(value, vmin=0.0, vmax=1.0):
    """Return a CSS colour interpolated from cool blue to hot red."""
    t = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1e-9)))
    r = int(30  + t * 220)
    g = int(100 - t * 70)
    b = int(220 - t * 190)
    return f"rgb({r},{g},{b})"


def colour_for(label):
    return CONTROLLER_COLORS.get(label, ACCENT)


# ----------------------------------------------------- Streamlit version shim
try:
    _VER = tuple(int(p) for p in st.__version__.split(".")[:2])
except Exception:                                      # pragma: no cover
    _VER = (0, 0)
_NEW_API = _VER >= (1, 46)


def fit(fn, *args, **kwargs):
    """Render a Streamlit element full width on any Streamlit version."""
    attempts = [{"width": "stretch"}, {"use_container_width": True},
                {"use_column_width": True}, {}]
    if not _NEW_API:
        attempts = attempts[1:] + attempts[:1]
    last = None
    for extra in attempts:
        try:
            return fn(*args, **kwargs, **extra)
        except TypeError as exc:
            last = exc
    raise last


# -------------------------------------------------------------------- CSS
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Outfit:wght@700;800&display=swap');

/* -------- base -------- */
html, body, .stApp, [class*="css"] {
  font-family: 'Inter', -apple-system, Segoe UI, sans-serif;
  color: #c5cde8;
  background: #0a0f1e;
}
.stApp { background: #0a0f1e; }
.block-container {
  padding-top: 1rem;
  padding-bottom: 3rem;
  max-width: 1560px;
}
h1,h2,h3,h4 {
  color: #f0f4ff;
  font-family: 'Outfit', 'Inter', sans-serif;
  font-weight: 750;
  letter-spacing: -0.02em;
}
[data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer { display: none; }

/* -------- animated gradient hero -------- */
@keyframes heroShimmer {
  0%   { background-position: 0%   50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0%   50%; }
}
.topbar {
  background: linear-gradient(135deg, #0a0f1e 0%, #0d1b35 40%, #0f2557 70%, #0a1931 100%);
  background-size: 300% 300%;
  animation: heroShimmer 8s ease infinite;
  border-radius: 14px;
  padding: 22px 28px;
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 24px;
  border: 1px solid #1e3a6a;
  border-bottom: 3px solid #3b82f6;
  position: relative;
  overflow: hidden;
}
.topbar::before {
  content: '';
  position: absolute;
  top: -60%; left: -30%;
  width: 60%; height: 200%;
  background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.topbar .title {
  color: #f0f4ff;
  font-family: 'Outfit', sans-serif;
  font-size: 1.55rem;
  font-weight: 800;
  letter-spacing: -0.025em;
  background: linear-gradient(90deg, #f0f4ff 0%, #93c5fd 50%, #818cf8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.topbar .sub {
  color: #8fa8cc;
  font-size: .88rem;
  margin-top: 6px;
  max-width: 72rem;
  line-height: 1.5;
}
.topbar .meta {
  color: #6b7fa8;
  font-size: .76rem;
  text-align: right;
  line-height: 1.8;
  white-space: nowrap;
}
.topbar .meta b { color: #93c5fd; font-weight: 600; }

/* -------- section headings -------- */
.eyebrow {
  font-size: .68rem;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: #3b82f6;
  font-weight: 700;
  margin-bottom: 3px;
}
.sec-title { font-size: 1.08rem; font-weight: 700; color: #f0f4ff; }
.sec-sub   { font-size: .84rem; color: #6b7fa8; margin-top: 3px; }
.sec {
  border-top: 1px solid #1e2d4a;
  padding-top: 16px;
  margin: 24px 0 12px;
}

/* -------- glassmorphism cards -------- */
.card {
  background: rgba(17,24,39,0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid #1e2d4a;
  border-radius: 14px;
  padding: 18px 20px;
  height: 100%;
  transition: border-color .2s, box-shadow .2s;
}
.card:hover {
  border-color: #2a4a8a;
  box-shadow: 0 0 28px rgba(59,130,246,0.08);
}

/* -------- KPI cards -------- */
.kpi .label {
  font-size: .68rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: #6b7fa8;
  font-weight: 700;
}
.kpi .value {
  font-size: 2.1rem;
  font-weight: 750;
  color: #f0f4ff;
  line-height: 1.1;
  margin-top: 8px;
  font-variant-numeric: tabular-nums;
  font-family: 'Outfit', sans-serif;
}
.kpi .sub { font-size: .78rem; color: #6b7fa8; margin-top: 5px; }
.kpi .delta {
  font-size: .8rem;
  font-weight: 700;
  margin-top: 10px;
  display: inline-block;
  padding: 3px 10px;
  border-radius: 6px;
}
.delta.good { color: #4ade80; background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.25); }
.delta.bad  { color: #f87171; background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.25); }
.delta.flat { color: #94a3b8; background: rgba(148,163,184,0.1); border: 1px solid rgba(148,163,184,0.2); }

/* -------- badges -------- */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.6; }
}
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 6px;
  font-size: .72rem;
  font-weight: 700;
  border: 1px solid transparent;
  letter-spacing: .03em;
}
.badge-green { background: rgba(34,197,94,0.12);  color: #4ade80; border-color: rgba(34,197,94,0.3); }
.badge-amber { background: rgba(245,158,11,0.12); color: #fbbf24; border-color: rgba(245,158,11,0.3); }
.badge-red   { background: rgba(239,68,68,0.12);  color: #f87171; border-color: rgba(239,68,68,0.3); }
.badge-blue  { background: rgba(59,130,246,0.12); color: #60a5fa; border-color: rgba(59,130,246,0.3); }
.badge-grey  { background: rgba(148,163,184,0.1); color: #94a3b8; border-color: rgba(148,163,184,0.2); }
.badge-violet{ background: rgba(168,85,247,0.12); color: #c084fc; border-color: rgba(168,85,247,0.3); }
.badge-live  { animation: pulse 1.8s ease-in-out infinite; }

/* -------- intersection cards -------- */
.tl-card {
  background: rgba(17,24,39,0.8);
  backdrop-filter: blur(8px);
  border: 1px solid #1e2d4a;
  border-radius: 12px;
  padding: 14px 16px;
  transition: border-color .2s;
}
.tl-card:hover { border-color: #2a4a8a; }
.tl-card .id {
  font-size: .66rem;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: #3b82f6;
  font-weight: 700;
}
.tl-card .phase { font-size: .95rem; color: #f0f4ff; font-weight: 700; margin: 4px 0 8px; }
.reason {
  font-size: .76rem;
  border-radius: 8px;
  padding: 8px 11px;
  margin-top: 8px;
  line-height: 1.45;
  border: 1px solid #1e2d4a;
  background: rgba(26,34,54,0.8);
  color: #8fa8cc;
}
.reason-rl         { background: rgba(59,130,246,0.08);  border-color: rgba(59,130,246,0.25); color: #93c5fd; }
.reason-emergency  { background: rgba(239,68,68,0.08);   border-color: rgba(239,68,68,0.25);  color: #fca5a5; }
.reason-pedestrian { background: rgba(245,158,11,0.08);  border-color: rgba(245,158,11,0.25); color: #fcd34d; }
.reason code, .mono { font-family: 'IBM Plex Mono', monospace; }

/* -------- pipeline flow -------- */
.pipe {
  display: flex;
  gap: 0;
  flex-wrap: wrap;
  border: 1px solid #1e2d4a;
  border-radius: 14px;
  background: rgba(17,24,39,0.6);
  backdrop-filter: blur(8px);
  overflow: hidden;
}
.pipe .node {
  flex: 1 1 160px;
  min-width: 160px;
  padding: 18px 18px;
  border-right: 1px solid #1e2d4a;
  position: relative;
  transition: background .2s;
}
.pipe .node:hover { background: rgba(59,130,246,0.05); }
.pipe .node:last-child { border-right: none; }
.pipe .node::after {
  content: '›';
  position: absolute;
  right: -10px;
  top: 50%;
  transform: translateY(-50%);
  color: #3b82f6;
  font-size: 1.4rem;
  font-weight: 700;
  z-index: 1;
}
.pipe .node:last-child::after { display: none; }
.pipe .n  { font-size: .66rem; letter-spacing: .14em; color: #3b82f6; font-weight: 800; }
.pipe .t  { font-weight: 700; font-size: .95rem; color: #f0f4ff; margin-top: 4px; }
.pipe .d  { font-size: .76rem; color: #6b7fa8; margin-top: 6px; line-height: 1.5; }

/* -------- tabs -------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px;
  border-bottom: 1px solid #1e2d4a;
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  height: 42px;
  padding: 0 20px;
  font-weight: 600;
  font-size: .88rem;
  color: #6b7fa8;
  background: transparent;
  border-radius: 8px 8px 0 0;
  transition: color .15s, background .15s;
}
.stTabs [data-baseweb="tab"]:hover { color: #c5cde8; background: rgba(59,130,246,0.05); }
.stTabs [aria-selected="true"] {
  color: #f0f4ff !important;
  border-bottom: 2px solid #3b82f6 !important;
  background: rgba(59,130,246,0.07) !important;
}

/* -------- sidebar -------- */
[data-testid="stSidebar"] { background: #0d1425; border-right: 1px solid #1e2d4a; }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f0f4ff; }

/* -------- metrics -------- */
[data-testid="stMetricValue"] {
  font-size: 1.5rem;
  font-variant-numeric: tabular-nums;
  color: #f0f4ff;
  font-weight: 700;
}
[data-testid="stMetricLabel"] { color: #6b7fa8; font-size: .8rem; }
[data-testid="stMetricDelta"] { font-size: .82rem; }

/* -------- inputs and selects -------- */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: #111827 !important;
  border-color: #1e2d4a !important;
  color: #c5cde8 !important;
}

/* -------- tables -------- */
[data-testid="stDataFrame"] {
  border: 1px solid #1e2d4a !important;
  border-radius: 10px;
  overflow: hidden;
}

/* -------- note / callout -------- */
.note {
  background: rgba(59,130,246,0.06);
  border: 1px solid rgba(59,130,246,0.25);
  border-left: 3px solid #3b82f6;
  border-radius: 10px;
  padding: 14px 16px;
  font-size: .84rem;
  color: #8fa8cc;
  line-height: 1.5;
}

/* -------- team member card -------- */
.team-card {
  background: rgba(17,24,39,0.7);
  border: 1px solid #1e2d4a;
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 8px;
  transition: border-color .2s;
}
.team-card:hover { border-color: #3b82f6; }
.team-card .member-name { font-weight: 700; color: #f0f4ff; font-size: .88rem; }
.team-card .member-module {
  font-size: .7rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #3b82f6;
  font-weight: 700;
  margin-bottom: 2px;
}
.team-card .member-role { font-size: .75rem; color: #6b7fa8; margin-top: 2px; }

/* -------- radar / chart containers -------- */
.chart-glass {
  background: rgba(17,24,39,0.6);
  backdrop-filter: blur(10px);
  border: 1px solid #1e2d4a;
  border-radius: 14px;
  padding: 4px;
}

/* -------- misc -------- */
.small { font-size: .8rem; color: #6b7fa8; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def topbar(title, subtitle, meta_lines=()):
    meta = "<br>".join(meta_lines)
    st.markdown(f"""<div class="topbar">
      <div><div class="title">{title}</div><div class="sub">{subtitle}</div></div>
      <div class="meta">{meta}</div></div>""", unsafe_allow_html=True)


def kpi(label, value, sub="", delta=None, good_down=True):
    d = ""
    if delta is not None:
        good = (delta < 0) if good_down else (delta > 0)
        cls = "flat" if abs(delta) < 0.05 else ("good" if good else "bad")
        arrow = "↓" if delta < 0 else "↑"
        d = f"<div class='delta {cls}'>{arrow} {abs(delta):.1f}%</div>"
    st.markdown(f"""<div class="card kpi"><div class="label">{label}</div>
        <div class="value">{value}</div><div class="sub">{sub}</div>{d}</div>""",
                unsafe_allow_html=True)


def section(title, subtitle="", eyebrow=""):
    st.markdown(f"""<div class="sec">{f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ''}
        <div class="sec-title">{title}</div>
        {f'<div class="sec-sub">{subtitle}</div>' if subtitle else ''}</div>""",
                unsafe_allow_html=True)


def note(text):
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def team_card(module, name, role):
    st.markdown(f"""<div class="team-card">
      <div class="member-module">{module}</div>
      <div class="member-name">{name}</div>
      <div class="member-role">{role}</div>
    </div>""", unsafe_allow_html=True)


def style_fig(fig, height=380, legend=True, title=None):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=13.5, color=INK), x=0, xanchor="left") if title else None,
        font=dict(family="Inter, sans-serif", size=12, color=BODY),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=42 if title else 16, b=8),
        showlegend=legend,
        legend=dict(
            orientation="h", y=1.1, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=MUTED),
        ),
        hoverlabel=dict(
            font_family="Inter, sans-serif",
            bgcolor="#0d1b35",
            font_color="#f0f4ff",
            bordercolor="#3b82f6",
        ),
    )
    axis = dict(
        gridcolor="#1a2a45",
        zerolinecolor="#1e2d4a",
        linecolor=BORDER,
        tickfont=dict(size=11, color=MUTED),
        title_font=dict(size=11.5, color=MUTED),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig

# ui_theme.py
import streamlit as st

def init_theme(show_toggle: bool = False):
    """
    Inicializa/aplica un tema GLOBAL en modo oscuro (forzado).
    Mantiene la misma API que antes para no romper otras páginas.
    """
    # Forzamos modo oscuro sin toggle
    t = {
        "template": "plotly_dark",
        "bg": "#0e1117",       # fondo app
        "text": "#e5e7eb",     # texto principal
        "accent": "#60a5fa",   # color acento (botones, toggles, etc.)
        "sidebar_bg": "#111111",
        "sidebar_text": "#e5e7eb",
        "header_bg": "#1e1e1e",
        "map_style": "carto-darkmatter",
        # Controles
        "input_bg": "#171a21",
        "input_border": "#374151",
        "input_text": "#e5e7eb",
        # Tabla
        "table_header_bg": "rgba(255,255,255,0.06)",
        "table_row_bg": "#0f1117",
    }

    st.session_state["_theme_mode"] = "dark"
    st.session_state["_theme"] = t

    # ===== CSS GLOBAL (oscuro) =====
    st.markdown(
        f"""
        <style>
        /* Fondo general y texto */
        .stApp {{
            background-color: {t['bg']};
            color: {t['text']};
        }}

        /* Header */
        header, [data-testid="stHeader"] {{
            background-color: {t['header_bg']} !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {t['sidebar_bg']} !important;
            color: {t['sidebar_text']} !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: {t['sidebar_text']} !important;
        }}

        /* Controles (select/multiselect/inputs) */
        section[data-testid="stSidebar"] div[ data-baseweb="select" ] > div,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {{
            background: {t['input_bg']} !important;
            color: {t['input_text']} !important;
            border: 1px solid {t['input_border']} !important;
            border-radius: 8px !important;
        }}
        section[data-testid="stSidebar"] div[ data-baseweb="select" ] * {{
            color: {t['input_text']} !important;
        }}

        /* Botones y descarga */
        .stButton>button, .stDownloadButton>button {{
            background-color: {t['accent']} !important;
            color: white !important;
            border: none !important;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            background-color: {t['accent']}cc !important;
        }}

        /* KPIs (st.metric) */
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricValue"] * {{
            color: {t['text']} !important;
        }}

        /* DataFrame (tabla) */
        div[data-testid="stDataFrame"] {{
            background: {t['bg']} !important;
            color: {t['text']} !important;
            border-radius: 8px;
        }}
        div[data-testid="stDataFrame"] thead tr th {{
            background: {t['table_header_bg']} !important;
            color: {t['text']} !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stDataFrame"] tbody tr td {{
            background: {t['table_row_bg']} !important;
            color: {t['text']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    return t


def style_fig(fig):
    """
    Aplica tema Plotly oscuro a la figura y refuerza colores de texto/ejes.
    """
    t = st.session_state.get("_theme", {
        "template": "plotly_dark", "bg": "#0e1117", "text": "#e5e7eb"
    })
    fig.update_layout(
        template=t["template"],
        paper_bgcolor=t["bg"],
        plot_bgcolor=t["bg"],
        font_color=t["text"],
        title_font_color=t["text"],
        legend=dict(font=dict(color=t["text"])),
        xaxis=dict(
            title_font_color=t["text"],
            tickfont=dict(color=t["text"]),
            gridcolor="rgba(255,255,255,0.15)"
        ),
        yaxis=dict(
            title_font_color=t["text"],
            tickfont=dict(color=t["text"]),
            gridcolor="rgba(255,255,255,0.15)"
        ),
    )
    return fig


def map_style() -> str:
    """Estilo de Mapbox (oscuro)."""
    return st.session_state.get("_theme", {}).get("map_style", "carto-darkmatter")

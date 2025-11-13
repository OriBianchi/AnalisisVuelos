# -*- coding: utf-8 -*-
# pages/1_Rutas_y_Aerolineas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import unicodedata, sys, textwrap
from PIL import Image
import base64

st.set_page_config(page_title="Vuelos y Aeropuertos", layout="wide")


# =========================
#  LOGO SIDEBAR
# =========================
def _uade_sidebar_header():
    logo_path = Path("assets/logos/uade.png")
    if not logo_path.exists():
        return

    # Logo en base64 para usarlo en CSS
    with open(logo_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""
        <style>
        /* Logo arriba del todo */
        [data-testid="stSidebarNav"] {{
            background-image: url("data:image/png;base64,{img_b64}");
            background-repeat: no-repeat;
            background-position: 16px 8px;
            background-size: 120px auto;
            padding-top: 75px;
        }}

        /* Texto del grupo */
        [data-testid="stSidebarNav"]::before {{
            white-space: pre-line;
            display: block;
            margin: 0px 0px 60px 0px;
            padding-left: 10px;
            border-left: 1px solid rgba(255,255,255,0.35);
            font-size: 12px;
            color: #e5e7eb;
            content:
                "Grupo 1 - Ciencia de Datos\\A"
                "Reinaldo Barreto\\A"
                "Oriana Bianchi\\A"
                "Federico Rey\\A"
                "Gabriel Kraus";
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_uade_sidebar_header()


# =========================
#  TEMA GLOBAL
# =========================
try:
    from ui_theme import init_theme, style_fig, map_style, show_global_title
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from ui_theme import init_theme, style_fig, map_style, show_global_title

init_theme(show_toggle=False)
show_global_title()

st.markdown(
    """
    <div style="
        width: 100%;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 25px;
        padding: 14px 18px;
        background-color: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        color: #e5e7eb;
        font-size: 16px;
        line-height: 1.45;
    ">
    Análisis del comportamiento aéreo local en Argentina, en base a datos de vuelos comerciales provistos por la ANAC.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
#  UI HELPERS
# =========================
def _theme_vals():
    return st.session_state.get(
        "_theme",
        {"bg": "#0e1117", "text": "#e5e7eb", "accent": "#60a5fa"},
    )


def _big_titles(fig, size=24):
    fig.update_layout(title=dict(font=dict(size=size)))
    return fig


def _force_axes_colors(fig):
    t = _theme_vals()
    fig.update_layout(
        font=dict(color=t["text"]),
        title_font_color=t["text"],
        xaxis=dict(title_font_color=t["text"], tickfont=dict(color=t["text"])),
        yaxis=dict(title_font_color=t["text"], tickfont=dict(color=t["text"])),
    )
    return fig


def _patch_light_css():
    t = _theme_vals()
    st.markdown(
        f"""
        <style>
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricDelta"] {{
            color: {t['text']} !important;
        }}

        div[data-testid="stDataFrame"] {{
            background: {t['bg']} !important;
            color: {t['text']} !important;
            border-radius: 8px;
        }}

        div[data-testid="stDataFrame"] * {{
            color: {t['text']} !important;
        }}

        div[data-testid="stDataFrame"] thead tr th {{
            background: rgba(255,255,255,0.04) !important;
            font-weight: 600 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_patch_light_css()


# =========================
#  FILE HELPERS
# =========================
def _find_data_dir(this_file: Path) -> Path:
    candidates = [this_file.parent / "data", this_file.parent.parent / "data"]
    for c in candidates:
        if c.exists():
            return c
    return this_file.parent / "data"


def _std_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.upper()


def _assets_dir(this_file: Path) -> Path:
    candidates = [
        this_file.parent / "assets" / "logos",
        this_file.parent.parent / "assets" / "logos",
    ]
    for c in candidates:
        if c.exists():
            return c
    return this_file.parent / "assets" / "logos"


def _icons_dir(this_file: Path) -> Path:
    candidates = [
        this_file.parent / "assets" / "icons",
        this_file.parent.parent / "assets" / "icons",
    ]
    for c in candidates:
        if c.exists():
            return c
    return this_file.parent / "assets" / "icons"


def _normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.upper().split())


_LOGO_MAP = {
    "AEROLINEAS ARGENTINAS SA": "aerolineas_argentinas.png",
    "AEROLINEAS ARGENTINAS": "aerolineas_argentinas.png",
    "FB LINEAS AEREAS - FLYBONDI": "flybondi.png",
    "FLYBONDI": "flybondi.png",
    "JETSMART AIRLINES S.A.": "jetsmart.png",
    "JETSMART AIRLINES SA": "jetsmart.png",
    "JETSMART": "jetsmart.png",
}


def _logo_path_for_airline(raw_name: str, assets_dir: Path) -> Path | None:
    n = _normalize_name(raw_name)

    if n in _LOGO_MAP:
        p = assets_dir / _LOGO_MAP[n]
        return p if p.exists() else None

    if "FLYBONDI" in n:
        p = assets_dir / "flybondi.png"
        return p if p.exists() else None

    if "JETSMART" in n:
        p = assets_dir / "jetsmart.png"
        return p if p.exists() else None

    if "AEROLINEAS" in n and "ARGENTINAS" in n:
        p = assets_dir / "aerolineas_argentinas.png"
        return p if p.exists() else None

    return None


# =========================
#  LOAD DATA
# =========================
DATA_DIR = _find_data_dir(Path(__file__).resolve())


@st.cache_data(show_spinner=True)
def load_data_unificado(data_dir: Path):
    """
    Lee el dataset unificado (vuelos).
    Espera columnas:
    Fecha, Fecha_Hora_Local_despegue, Origen, Destino, Aerolinea,
    Aeronave, Pasajeros, Vuelo_ID, Fecha_Hora_Local_aterrizaje,
    Tiempo_Vuelo
    """

    CANDIDATES = ["vuelos_preliminar.csv"]
    path = None

    for name in CANDIDATES:
        p = data_dir / name
        if p.exists():
            path = p
            break

    if path is None:
        raise FileNotFoundError(
            f"No encontré un CSV unificado en {data_dir}. "
            f"Colocá tu archivo como 'vuelos_unificado.csv'."
        )

    df = pd.read_csv(
        path,
        low_memory=False,
        parse_dates=[
            "Fecha",
            "Fecha_Hora_Local_despegue",
            "Fecha_Hora_Local_aterrizaje",
        ],
    )

    # Normalizamos texto
    for c in [
        "Origen",
        "Destino",
        "Aerolinea",
        "Aeronave",
        "Ciudad_O",
        "Provincia_O",
        "Ciudad_D",
        "Provincia_D",
    ]:
        if c in df.columns:
            df[c] = _std_text(df[c])

    # Derivados
    df["AÑO"] = df["Fecha"].dt.year
    df["MES"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()

    df["PAX"] = (
        pd.to_numeric(df["Pasajeros"], errors="coerce")
        if "Pasajeros" in df.columns
        else np.nan
    )

    # Tiempo de vuelo en minutos
    if "Tiempo_Vuelo" in df.columns:
        try:
            t = pd.to_timedelta(df["Tiempo_Vuelo"])
            df["MINUTOS_VUELO"] = (
                t.dt.total_seconds() / 60.0
            ).round().astype("Int64")
        except Exception:
            df["MINUTOS_VUELO"] = pd.NA
    else:
        df["MINUTOS_VUELO"] = pd.NA

    df["AEROLINEA"] = df["Aerolinea"]

    return df


@st.cache_data
def _airport_code_to_name(data_dir: Path):
    p = data_dir / "aeropuertos_detalle.csv"
    if not p.exists():
        return {}

    det = pd.read_csv(p, sep=";", dtype=str).fillna("")

    for c in ["local", "oaci", "iata", "denominacion"]:
        if c in det.columns:
            det[c] = det[c].astype(str).str.strip()

    for c in ["local", "oaci", "iata"]:
        if c in det.columns:
            det[c] = det[c].str.upper()

    m = pd.melt(
        det,
        id_vars=["denominacion"],
        value_vars=[c for c in ["local", "iata", "oaci"] if c in det.columns],
        var_name="tipo",
        value_name="code",
    )

    m = m[m["code"].ne("")]
    return m.drop_duplicates("code").set_index("code")["denominacion"].to_dict()


df_all = load_data_unificado(DATA_DIR)
code_to_name = _airport_code_to_name(DATA_DIR)


# =========================
#  SIDEBAR FILTROS
# =========================
st.sidebar.header("Filtros")


def _apply(df, col, values, exclude=False):
    if not values:
        return df
    return df[~df[col].isin(values)] if exclude else df[df[col].isin(values)]


years = sorted(df_all["AÑO"].dropna().unique().tolist())
aerolineas = sorted(df_all["AEROLINEA"].dropna().unique().tolist())
all_airports = sorted(
    pd.Index(df_all["Origen"].dropna().unique())
    .union(df_all["Destino"].dropna().unique())
    .tolist()
)

airport_labels = [
    f"{c} — {code_to_name.get(c, c)}" for c in all_airports
]

code_from_label = {lab: lab.split(" — ", 1)[0] for lab in airport_labels}


# ===== Año =====
with st.sidebar.container():
    st.caption("Año")
    sel_years = st.multiselect("Seleccioná años", years, key="year_ms")
    excl_years = st.checkbox("Excluir selección", key="year_excl", value=False)

# ===== Mes =====
with st.sidebar.container():
    st.caption("Mes")
    _meses = {
        1: "Enero",
        2: "Febrero",
        3: "Marzo",
        4: "Abril",
        5: "Mayo",
        6: "Junio",
        7: "Julio",
        8: "Agosto",
        9: "Septiembre",
        10: "Octubre",
        11: "Noviembre",
        12: "Diciembre",
    }
    month_labels = [_meses[m] for m in range(1, 13)]
    label_to_num = {v: k for k, v in _meses.items()}

    sel_month_labels = st.multiselect(
        "Seleccioná meses",
        options=month_labels,
        key="month_ms",
    )

    sel_months = [label_to_num[l] for l in sel_month_labels]
    excl_month = st.checkbox("Excluir selección", key="month_excl", value=False)

# ===== Aerolínea =====
with st.sidebar.container():
    st.caption("Aerolínea")
    sel_aerolinea = st.multiselect(
        "Seleccioná aerolíneas",
        aerolineas,
        key="airline_ms",
    )
    excl_airline = st.checkbox("Excluir selección", key="airline_excl", value=False)


# =========================
#  APLICAR FILTROS
# =========================
df = df_all.copy()

df = _apply(df, "AÑO", sel_years, exclude=excl_years)

if sel_months:
    df = (
        df[~df["Fecha"].dt.month.isin(sel_months)]
        if excl_month
        else df[df["Fecha"].dt.month.isin(sel_months)]
    )

df = _apply(df, "AEROLINEA", sel_aerolinea, exclude=excl_airline)


# =========================
#  FILTROS ORIGEN/DESTINO
# =========================
if "filter_origen" in st.session_state:
    sel_origen = st.session_state["filter_origen"]
else:
    sel_origen = []

if "filter_destino" in st.session_state:
    sel_destino = st.session_state["filter_destino"]
else:
    sel_destino = []

if sel_origen:
    df = df[df["Origen"].isin(sel_origen)]
if sel_destino:
    df = df[df["Destino"].isin(sel_destino)]


# =========================
#  KPIs
# =========================
st.markdown(
    """
    <style>
    div[data-testid="stMetricLabel"] { display:none !important; }
    .kpi-title{
        font-size:28px;
        font-weight:400;
        color:#e5e7eb;
        line-height:1.15;
        margin:0 0 -6px 0;
    }
    div[data-testid="stMetricValue"]{
        font-size:34px !important;
        font-weight:800 !important;
        color:#ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

total_vuelos = (
    int(df["Vuelo_ID"].nunique())
    if "Vuelo_ID" in df.columns
    else len(df)
)

aerolineas_activas = int(df["AEROLINEA"].nunique())

aeropuertos_activos = int(
    pd.Index(df["Origen"].dropna().unique())
    .union(df["Destino"].dropna().unique())
    .nunique()
)

if not df.empty:
    ruta_top = (
        df.groupby(["Origen", "Destino"])["Vuelo_ID"]
        .nunique()
        .reset_index()
        .sort_values("Vuelo_ID", ascending=False)
        .head(1)
    )
    ruta_nombre = f"{ruta_top['Origen'].iloc[0]} → {ruta_top['Destino'].iloc[0]}"
    ruta_vuelos = int(ruta_top["Vuelo_ID"].iloc[0])
else:
    ruta_nombre = "—"
    ruta_vuelos = 0

if not df.empty:
    df["_MES_NUM"] = df["Fecha"].dt.month
    mes_top = (
        df.groupby("_MES_NUM")["Vuelo_ID"]
        .nunique()
        .reset_index()
        .sort_values("Vuelo_ID", ascending=False)
        .head(1)
    )
    meses_txt = _meses
    mes_nombre = meses_txt[int(mes_top["_MES_NUM"].iloc[0])]
    mes_vuelos = int(mes_top["Vuelo_ID"].iloc[0])
else:
    mes_nombre = "—"
    mes_vuelos = 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="kpi-title">✈️ Vuelos</div>', unsafe_allow_html=True)
    st.metric("", f"{total_vuelos:,}")

with c2:
    st.markdown('<div class="kpi-title">🏆 Ruta más operada</div>', unsafe_allow_html=True)
    st.metric("", ruta_nombre, f"{ruta_vuelos:,} vuelos")

with c3:
    st.markdown('<div class="kpi-title">🗓️ Mes con más movimiento</div>', unsafe_allow_html=True)
    st.metric("", mes_nombre, f"{mes_vuelos:,} vuelos")

with c4:
    st.markdown('<div class="kpi-title">🛫 Aeropuertos</div>', unsafe_allow_html=True)
    st.metric("", f"{aeropuertos_activos:,}")

st.markdown("---")


# =========================
#  GRÁFICO VUELOS POR AÑO
# =========================
ICONS_DIR = _icons_dir(Path(__file__).resolve())
ICON_PATH = ICONS_DIR / "plane.png"

by_year = (
    df.groupby("AÑO").agg(
        VUELOS=('Vuelo_ID', 'nunique')
        if "Vuelo_ID" in df.columns
        else ('Vuelo_ID', 'size')
    )
)

by_year = by_year.reset_index().sort_values("VUELOS", ascending=True)

fig_year = px.bar(
    by_year,
    x="VUELOS",
    y="AÑO",
    orientation="h",
    title="✈️ Vuelos por año",
    text_auto=True,
)

fig_year.update_traces(marker_color="#7ec8e3")
fig_year.update_layout(
    xaxis_title="Vuelos",
    yaxis_title="Año",
    yaxis=dict(categoryorder="total ascending"),
)

if ICON_PATH.exists() and not by_year.empty:
    try:
        plane_img = Image.open(ICON_PATH)
        x_max = float(by_year["VUELOS"].max())
        pad = 0.2 * x_max
        fig_year.update_xaxes(range=[0, x_max + pad])

        size_x, size_y = 2 * x_max, 2

        for _, r in by_year.iterrows():
            fig_year.add_layout_image(
                dict(
                    source=plane_img,
                    xref="x",
                    yref="y",
                    x=float(r["VUELOS"]),
                    y=r["AÑO"],
                    sizex=size_x,
                    sizey=size_y,
                    xanchor="left",
                    yanchor="middle",
                    layer="above",
                )
            )
    except Exception:
        pass

fig_year = style_fig(fig_year)
fig_year = _force_axes_colors(fig_year)
fig_year = _big_titles(fig_year, size=24)

st.plotly_chart(fig_year, use_container_width=True)
st.markdown("---")


# =========================
#  GRÁFICO VUELOS POR MES
# =========================
st.markdown("### 🗓️✈️ Vuelos por mes")

if not df.empty:
    tmp = df.copy()
    tmp["_MES_NUM"] = tmp["Fecha"].dt.month

    by_month_cal = (
        tmp.groupby("_MES_NUM")["Vuelo_ID"]
        .nunique()
        .reindex(range(1, 13), fill_value=0)
        .reset_index(name="VUELOS")
    )

    by_month_cal["MES"] = pd.Categorical(
        by_month_cal["_MES_NUM"].map(_meses),
        categories=[_meses[i] for i in range(1, 13)],
        ordered=True,
    )

    total_all = int(by_month_cal["VUELOS"].sum())
    by_month_cal["PCT"] = (by_month_cal["VUELOS"] / total_all).fillna(0)

    # Mes top
    max_vuelos = by_month_cal["VUELOS"].max()
    mes_top_num = int(by_month_cal.loc[by_month_cal["VUELOS"].idxmax(), "_MES_NUM"])
    mes_top_nombre = _meses[mes_top_num]

    colors = [
        "#fdd835" if v == max_vuelos else "#60a5fa"
        for v in by_month_cal["VUELOS"]
    ]

    fig_month_tot = px.bar(
        by_month_cal.sort_values("MES"),
        x="MES",
        y="VUELOS",
        text=by_month_cal["VUELOS"].map(lambda x: f"{x:,}"),
    )

    fig_month_tot.update_traces(
        marker_color=colors,
        insidetextanchor="middle",
        textfont=dict(size=16, color="#0e1117", weight="bold"),
        cliponaxis=False,
        hovertemplate=(
            "🗓️ <b>%{x}</b><br>"
            "✈️ Vuelos: %{y:,}<br>"
            "% participación: %{customdata:.1%}<extra></extra>"
        ),
        customdata=by_month_cal["PCT"].to_numpy(),
    )

    fig_month_tot.add_annotation(
        x=mes_top_nombre,
        y=max_vuelos * 1.05,
        text="👑",
        showarrow=False,
        font=dict(size=32),
        xanchor="center",
    )

    fig_month_tot.update_layout(
        barcornerradius=6,
        xaxis_title="Mes",
        yaxis_title="Cantidad de vuelos",
        xaxis=dict(tickangle=0, showgrid=False),
        margin=dict(l=10, r=10, t=60, b=40),
        title=" ",
    )

    fig_month_tot = style_fig(fig_month_tot)
    fig_month_tot = _force_axes_colors(fig_month_tot)
    fig_month_tot = _big_titles(fig_month_tot, size=24)

    st.plotly_chart(fig_month_tot, use_container_width=True)
else:
    st.info("No hay datos para los filtros seleccionados.")

st.markdown("---")


# =========================
#  DONUT AEROLÍNEAS
# =========================
TOP_N = 10

by_airline = (
    df.groupby("AEROLINEA")["Vuelo_ID"]
    .nunique()
    .reset_index(name="VUELOS")
    .sort_values("VUELOS", ascending=False)
    .head(TOP_N)
)

by_airline["AEROLINEA_NORM"] = by_airline["AEROLINEA"].apply(_normalize_name)

custom_colors_norm = {
    "FB LINEAS AEREAS - FLYBONDI": "#fdbf0a",
    "FLYBONDI": "#fdbf0a",
    "AEROLINEAS ARGENTINAS SA": "#0080c5",
    "AEROLINEAS ARGENTINAS": "#0080c5",
    "JETSMART AIRLINES S.A.": "#981f2b",
    "JETSMART AIRLINES SA": "#981f2b",
    "JETSMART": "#981f2b",
}

default_color = "#bdbdbd"

pie_col, logos_col = st.columns([2.1, 1])

with pie_col:
    fig_airline = px.pie(
        by_airline,
        names="AEROLINEA",
        values="VUELOS",
        color="AEROLINEA_NORM",
        color_discrete_map=custom_colors_norm,
        title="Participación por Aerolínea",
        hole=0.35,
    )

    fig_airline.update_traces(
        textinfo="percent+label",
        textfont_size=16,
        insidetextorientation="radial",
        pull=[0.05] * len(by_airline),
        marker=dict(line=dict(width=0)),
    )

    fig_airline = style_fig(fig_airline)
    fig_airline = _force_axes_colors(fig_airline)
    fig_airline = _big_titles(fig_airline, size=24)

    fig_airline.update_layout(
        autosize=True,
        height=500,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=20),
    )

    st.plotly_chart(fig_airline, use_container_width=True)

assets_dir = _assets_dir(Path(__file__).resolve())

with logos_col:
    st.markdown("#### Detalle")
    total_vuelos_air = int(by_airline["VUELOS"].sum())

    for _, row in by_airline.iterrows():
        name = str(row["AEROLINEA"])
        name_norm = _normalize_name(name)
        vuelos = int(row["VUELOS"])
        pct = 100.0 * vuelos / total_vuelos_air if total_vuelos_air > 0 else 0.0
        color_txt = custom_colors_norm.get(name_norm, default_color)

        lc, rc = st.columns([1, 2.4])

        with lc:
            logo_path = _logo_path_for_airline(name, assets_dir)
            if logo_path:
                st.image(str(logo_path), use_container_width=True)
            else:
                st.markdown(
                    "<div style='width:100%;height:52px;border:1px dashed #555;"
                    "display:flex;align-items:center;justify-content:center;'>"
                    "<span style='font-size:11px;color:#aaa;'>Sin logo</span></div>",
                    unsafe_allow_html=True,
                )

        with rc:
            st.markdown(
                f"<span style='color:{color_txt};font-weight:700'>{name}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"{pct:.1f}% — {vuelos:,} vuelos")

st.markdown("---")

# =========================
#  TABLA TOP RUTAS
# =========================
def _pretty_with_code(code: str) -> str:
    su = str(code or "").upper().strip()
    full = code_to_name.get(su)
    return f"{full} ({su})" if full else su


def _wrap_html(txt: str, width: int = 28) -> str:
    try:
        return "<br>".join(textwrap.wrap(str(txt), width=width))
    except Exception:
        return str(txt)


# Inicializar session state
if "filter_origen" not in st.session_state:
    st.session_state["filter_origen"] = []

if "filter_destino" not in st.session_state:
    st.session_state["filter_destino"] = []

# ----- Filtros Top Conexiones -----
st.markdown("#### 🔍 Filtrar por aeropuerto")

valid_origenes = sorted(df["Origen"].dropna().unique().tolist())
valid_destinos = sorted(df["Destino"].dropna().unique().tolist())

if "filter_origen" in st.session_state:
    st.session_state["filter_origen"] = [
        o for o in st.session_state["filter_origen"] if o in valid_origenes
    ]

col_f1, col_f2 = st.columns(2)

with col_f1:
    sel_origen = st.multiselect(
        "Aeropuerto de ORIGEN",
        options=valid_origenes,
        format_func=lambda c: f"{c} — {code_to_name.get(c, c)}",
        key="filter_origen",
    )

if sel_origen:
    valid_destinos_filt = sorted(
        df[df["Origen"].isin(sel_origen)]["Destino"]
        .dropna()
        .unique()
        .tolist()
    )
else:
    valid_destinos_filt = valid_destinos

if "filter_destino" in st.session_state:
    st.session_state["filter_destino"] = [
        d for d in st.session_state["filter_destino"] if d in valid_destinos_filt
    ]

with col_f2:
    sel_destino = st.multiselect(
        "Aeropuerto de DESTINO",
        options=valid_destinos_filt,
        format_func=lambda c: f"{c} — {code_to_name.get(c, c)}",
        key="filter_destino",
    )

df_rutas_filter = df.copy()

if sel_origen:
    df_rutas_filter = df_rutas_filter[df_rutas_filter["Origen"].isin(sel_origen)]

if sel_destino:
    df_rutas_filter = df_rutas_filter[df_rutas_filter["Destino"].isin(sel_destino)]

if "Vuelo_ID" in df.columns:
    df_pax_unique = df_rutas_filter.groupby("Vuelo_ID")["PAX"].max()
    df_tmp = df_rutas_filter[["Vuelo_ID", "Origen", "Destino"]].drop_duplicates()
    df_tmp = df_tmp.merge(df_pax_unique, on="Vuelo_ID", how="left")

    rutas = (
        df_tmp.groupby(["Origen", "Destino"])
        .agg(VUELOS=("Vuelo_ID", "nunique"), PAX=("PAX", "sum"))
        .reset_index()
        .sort_values("VUELOS", ascending=False)
        .head(10)
    )
else:
    rutas = (
        df.groupby(["Origen", "Destino"])
        .agg(VUELOS=("Fecha", "size"), PAX=("PAX", "sum"))
        .reset_index()
        .sort_values("VUELOS", ascending=False)
        .head(10)
    )

rutas["ORIGEN_NICE"] = rutas["Origen"].apply(_pretty_with_code)
rutas["DESTINO_NICE"] = rutas["Destino"].apply(_pretty_with_code)

if rutas.empty:
    st.info("No hay rutas para los filtros actuales.")

else:
    n = len(rutas)
    fig_routes = go.Figure()

    fig_routes.update_xaxes(visible=False, range=[0, 1])
    fig_routes.update_yaxes(visible=False, range=[0, n + 1.8])

    base_color = "#60a5fa"
    accent_color = "#e5e7eb"

    band_a = "rgba(255,255,255,0.03)"
    band_b = "rgba(255,255,255,0.00)"

    X_NUM = 0.05
    X_ORIG = 0.18
    X_STATS = 0.36
    X_ARROW_A = 0.46
    X_ARROW_B = 0.74
    DEST_PANEL_L = 0.80
    DEST_PANEL_R = 0.97
    X_DEST = 0.88
    X_PLANE = 0.60
    Y_PLANE_OFF = 0.30

    # Panel destino
    fig_routes.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=DEST_PANEL_L,
        x1=DEST_PANEL_R,
        y0=0.2,
        y1=n + 1.2,
        line=dict(width=0),
    )

    # Bandas por fila
    for rank in range(1, n + 1):
        y = n - rank + 1
        color = band_a if rank % 2 else band_b

        fig_routes.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=0.10,
            x1=DEST_PANEL_R,
            y0=y - 0.48,
            y1=y + 0.48,
            line=dict(width=0),
            fillcolor=color,
        )

        fig_routes.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=0.10,
            x1=DEST_PANEL_R,
            y0=y - 0.5,
            y1=y - 0.5,
            line=dict(color="rgba(255,255,255,0.06)", width=1),
        )

    # Filas
    for rank, r in enumerate(rutas.itertuples(index=False), start=1):
        y = n - rank + 1

        origen = _wrap_html(r.ORIGEN_NICE, width=30)
        destino = _wrap_html(r.DESTINO_NICE, width=26)
        vuelos = int(r.VUELOS)
        pax = int(0 if pd.isna(r.PAX) else r.PAX)

        fig_routes.add_annotation(
            x=X_NUM,
            y=y,
            xref="x",
            yref="y",
            text=f"<b style='font-size:22px'>{rank}</b>",
            showarrow=False,
            font=dict(size=22, color=accent_color),
        )

        fig_routes.add_annotation(
            x=X_ORIG,
            y=y,
            xref="x",
            yref="y",
            text=origen,
            showarrow=False,
            align="left",
            font=dict(size=16, color=accent_color),
        )

        fig_routes.add_annotation(
            x=X_STATS,
            y=y,
            xref="x",
            yref="y",
            text=f"👥 PAX: <b>{pax:,}</b><br>✈️ Vuelos: <b>{vuelos:,}</b>",
            showarrow=False,
            align="center",
            font=dict(size=14, color=base_color),
        )

        fig_routes.add_annotation(
            x=X_ARROW_B,
            y=y,
            ax=X_ARROW_A,
            ay=y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor=base_color,
        )

        fig_routes.add_annotation(
            x=X_DEST,
            y=y,
            xref="x",
            yref="y",
            text=destino,
            showarrow=False,
            align="center",
            font=dict(size=16, color=accent_color),
        )

    # Avioncitos
    try:
        plane_icon = Image.open(ICONS_DIR / "plane.png")
        for rank in range(1, n + 1):
            y = n - rank + 1
            fig_routes.add_layout_image(
                dict(
                    source=plane_icon,
                    xref="x",
                    yref="y",
                    x=X_PLANE,
                    y=y + Y_PLANE_OFF,
                    sizex=0.18,
                    sizey=0.90,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                )
            )
    except Exception:
        pass

    fig_routes.update_layout(
        margin=dict(l=40, r=40, t=60, b=28),
        height=max(440, 67 * n),
        title=" ",
    )

    fig_routes = style_fig(fig_routes)
    fig_routes = _force_axes_colors(fig_routes)
    fig_routes = _big_titles(fig_routes, size=24)

    st.plotly_chart(fig_routes, use_container_width=True)

# -*- coding: utf-8 -*-
# pages/1_Rutas_y_Aerolineas.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import unicodedata
import sys
from PIL import Image

st.set_page_config(page_title="Vuelos y Aeropuertos", layout="wide")

# === Títulos grandes y KPI más visibles ===
st.markdown("""
<style>
:root{
  --title-xl: 24px;    /* tamaño para títulos de gráficos y h4 */
  --kpi-label: 20px;   /* etiqueta del KPI */
  --kpi-value: 28px;   /* valor del KPI */
}

/* KPI: etiqueta y valor */
div[data-testid="stMetricLabel"]{
  font-size: var(--kpi-label) !important;
  font-weight: 700 !important;
}
div[data-testid="stMetricValue"]{
  font-size: var(--kpi-value) !important;
}

/* H4 (#### Detalle) al mismo tamaño que los gráficos */
h4, .stMarkdown h4{
  font-size: var(--title-xl) !important;
}
</style>
""", unsafe_allow_html=True)

# === Import del tema global (ui_theme.py en la raíz) ===
try:
    from ui_theme import init_theme, style_fig, map_style
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from ui_theme import init_theme, style_fig, map_style

init_theme(show_toggle=False)  # modo oscuro fijo

# ========== SOLO LO NECESARIO PARA COLORES Y TABLA ==========
def _theme_vals():
    return st.session_state.get(
        "_theme",
        {"bg": "#0e1117", "text": "#e5e7eb", "accent": "#60a5fa"}
    )

def _big_titles(fig, size=24):
    """Aumenta el tamaño del título del gráfico."""
    fig.update_layout(title=dict(font=dict(size=size)))
    return fig

def _force_axes_colors(fig):
    """Fuerza colores de título, ejes y ticks según el tema actual."""
    t = _theme_vals()
    fig.update_layout(
        font=dict(color=t["text"]),
        title_font_color=t["text"],
        xaxis=dict(title_font_color=t["text"], tickfont=dict(color=t["text"])),
        yaxis=dict(title_font_color=t["text"], tickfont=dict(color=t["text"]))
    )
    return fig

# KPIs + DataFrame: asegurar colores correctos
def _patch_light_css():
    t = _theme_vals()
    st.markdown(
        f"""
        <style>
        /* KPIs */
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricDelta"] {{
            color: {t['text']} !important;
        }}

        /* DataFrame contenedor y tipografía */
        div[data-testid="stDataFrame"] {{
            background: {t['bg']} !important;
            color: {t['text']} !important;
            border-radius: 8px;
        }}
        div[data-testid="stDataFrame"] * {{
            color: {t['text']} !important;
        }}
        /* Cabeceras de la tabla */
        div[data-testid="stDataFrame"] thead tr th {{
            background: rgba(255,255,255,0.04) !important;
            font-weight: 600 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

_patch_light_css()
# ==========================================================================

# ---------- Helpers ----------
def _find_data_dir(this_file: Path) -> Path:
    candidates = [this_file.parent / "data", this_file.parent.parent / "data"]
    for c in candidates:
        if (c / "vuelos_eda.csv").exists():
            return c
    return this_file.parent / "data"

def _std_text(s):
    return s.fillna("").astype(str).str.strip().str.upper()

def _assets_dir(this_file: Path) -> Path:
    candidates = [this_file.parent / "assets" / "logos", this_file.parent.parent / "assets" / "logos"]
    for c in candidates:
        if c.exists():
            return c
    return this_file.parent / "assets" / "logos"

def _normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.upper().split())
    return s

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
        p = assets_dir / "flybondi.png";  return p if p.exists() else None
    if "JETSMART" in n:
        p = assets_dir / "jetsmart.png";  return p if p.exists() else None
    if "AEROLINEAS" in n and "ARGENTINAS" in n:
        p = assets_dir / "aerolineas_argentinas.png";  return p if p.exists() else None
    return None

@st.cache_data
def load_data():
    base = _find_data_dir(Path(__file__).resolve())
    eda = pd.read_csv(base / "vuelos_eda.csv", low_memory=False, parse_dates=["Fecha","Fecha_Hora_Local"])
    aero = pd.read_csv(base / "aeropuertos_eda.csv", low_memory=False)

    eda["AEROLINEA"] = (_std_text(eda["Aerolinea Nombre"])
                        if "Aerolinea Nombre" in eda.columns
                        else _std_text(eda.get("Aerolinea", pd.Series(index=eda.index))))
    eda["AEROPUERTO"] = _std_text(eda.get("Aeropuerto", pd.Series(index=eda.index)))
    eda["TIPO_MOV"] = _std_text(eda.get("Tipo Movimiento", pd.Series(index=eda.index)))
    eda["PAX"] = pd.to_numeric(eda.get("Pasajeros", np.nan), errors="coerce")
    eda["AÑO"] = eda["Fecha"].dt.year
    eda["MES"] = eda["Fecha"].dt.to_period("M").dt.to_timestamp()

    if "denominacion" in aero.columns:
        aero["DENOMINACION_STD"] = _std_text(aero["denominacion"])
    else:
        aero["DENOMINACION_STD"] = ""
    return eda, aero

eda, aero = load_data()

# === Soporte nombres de aeropuertos (labels bonitas) ===
@st.cache_data
def _airport_code_to_name():
    """Devuelve dict {codigo -> denominacion} usando aeropuertos_detalle.csv (sep=';')."""
    det_path = Path(__file__).resolve().parents[1] / "data" / "aeropuertos_detalle.csv"
    if not det_path.exists():
        return {}
    det = pd.read_csv(det_path, sep=";", dtype=str).fillna("")
    for c in ["local", "oaci", "iata", "denominacion"]:
        if c in det.columns:
            det[c] = det[c].astype(str).str.strip()
    for c in ["local", "oaci", "iata"]:
        if c in det.columns:
            det[c] = det[c].str.upper()
    m = pd.melt(det, id_vars=["denominacion"], value_vars=[c for c in ["local","iata","oaci"] if c in det.columns],
                var_name="tipo", value_name="code")
    m = m[m["code"].ne("")]
    return m.drop_duplicates("code").set_index("code")["denominacion"].to_dict()

code_to_name = _airport_code_to_name()

# ---------- Sidebar (Filtros) — ÚNICO BLOQUE, con keys ----------
st.sidebar.header("Filtros")

years = sorted(eda["AÑO"].dropna().unique().tolist())
sel_years = st.sidebar.multiselect("Año", years, default=years, key="year_ms")

aerolineas = sorted(eda["AEROLINEA"].dropna().unique().tolist())
sel_aerolinea = st.sidebar.multiselect("Aerolínea", aerolineas, key="airline_ms")

tipos = sorted(eda["TIPO_MOV"].dropna().unique().tolist())
sel_tipo = st.sidebar.multiselect("Tipo de movimiento", tipos, key="tipo_ms")

# Aeropuertos (muestra "CODIGO — Nombre completo")
all_codes = sorted(eda["AEROPUERTO"].dropna().unique().tolist())
airport_labels = [f"{c} — {code_to_name.get(c, c)}" for c in all_codes]
sel_airport_labels = st.sidebar.multiselect("Aeropuerto", options=airport_labels, key="airport_ms")
sel_airports = [s.split(" — ", 1)[0] for s in sel_airport_labels]

# Aplicación de filtros
df = eda.copy()
if sel_years:
    df = df[df["AÑO"].isin(sel_years)]
if sel_aerolinea:
    df = df[df["AEROLINEA"].isin(sel_aerolinea)]
if sel_tipo:
    df = df[df["TIPO_MOV"].isin(sel_tipo)]
if sel_airports:
    df = df[df["AEROPUERTO"].isin(sel_airports)]


# ---------- KPIs (título grande de verdad) ----------
st.markdown("""
<style>
/* Ocultamos el label pequeño nativo de st.metric */
div[data-testid="stMetricLabel"] { display: none !important; }

/* Título custom del KPI */
.kpi-title{
  font-size: 28px;        /* <-- ajustá aquí el tamaño del TÍTULO */
  font-weight: 400;
  color: #e5e7eb;         /* modo oscuro */
  line-height: 1.15;
  margin: 0 0 -6px 0;     /* pegado al valor */
}

/* Tamaño del valor del KPI (opcional) */
div[data-testid="stMetricValue"]{
  font-size: 34px !important;  /* <-- valor un toque más grande */
  font-weight: 800 !important;
  color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

total_vuelos = len(df)
total_pax = int(df["PAX"].sum()) if df["PAX"].notna().any() else None
aerolineas_activas = int(df["AEROLINEA"].nunique())
aeropuertos_activos = int(df["AEROPUERTO"].nunique())

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown('<div class="kpi-title">✈️ Vuelos</div>', unsafe_allow_html=True)
    st.metric(label="", value=f"{total_vuelos:,}")

with c2:
    st.markdown('<div class="kpi-title">🧍 PAX</div>', unsafe_allow_html=True)
    st.metric(label="", value=(f"{total_pax:,}" if total_pax is not None else "—"))

with c3:
    st.markdown('<div class="kpi-title">🏢 Aerolíneas</div>', unsafe_allow_html=True)
    st.metric(label="", value=f"{aerolineas_activas:,}")

with c4:
    st.markdown('<div class="kpi-title">🛫 Aeropuertos</div>', unsafe_allow_html=True)
    st.metric(label="", value=f"{aeropuertos_activos:,}")

st.markdown("---")

# ---------- Gráfico 1: Barras por año (con avioncitos al final) ----------
ICON_PATH = (Path(__file__).resolve().parents[1] / "assets" / "icons" / "plane.png")

by_year = (
    df.groupby("AÑO")
      .size()
      .reset_index(name="VUELOS")
      .sort_values("VUELOS", ascending=True)
)

fig_year = px.bar(
    by_year,
    x="VUELOS",
    y="AÑO",
    orientation="h",
    title="✈️ Cantidad de vuelos por año",
    text_auto=True
)
fig_year.update_traces(marker_color="#7ec8e3")
fig_year.update_layout(xaxis_title="Vuelos", yaxis_title="Año", yaxis=dict(categoryorder="total ascending"))

if ICON_PATH.exists():
    try:
        plane_img = Image.open(ICON_PATH)
        x_max = float(by_year["VUELOS"].max())
        pad   = 0.2 * x_max
        fig_year.update_xaxes(range=[0, x_max + pad])

        size_x = 2 * x_max
        size_y = 2
        for _, r in by_year.iterrows():
            x_val = float(r["VUELOS"])
            y_cat = r["AÑO"]
            fig_year.add_layout_image(dict(
                source=plane_img, xref="x", yref="y", x=x_val, y=y_cat,
                sizex=size_x, sizey=size_y, xanchor="left", yanchor="middle", layer="above"
            ))
    except Exception as e:
        st.warning(f"No se pudo cargar el ícono del avión: {e}")
else:
    st.warning("No se encontró el ícono del avión en assets/icons/plane.png")

fig_year = style_fig(fig_year)
fig_year = _force_axes_colors(fig_year)
fig_year = _big_titles(fig_year, size=24)          # << tamaño del título
st.plotly_chart(fig_year, use_container_width=True)

st.markdown("---")

# ---------- Gráfico 2: Donut por aerolínea + logos ----------
TOP_N = 10
by_airline = (
    df.groupby("AEROLINEA")
      .size()
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
        hole=0.35
    )
    fig_airline.update_traces(
        textinfo="percent+label",
        textfont_size=16,
        insidetextorientation="radial",
        pull=[0.05]*len(by_airline),
        marker=dict(line=dict(width=0))
    )
    fig_airline = style_fig(fig_airline)
    fig_airline = _force_axes_colors(fig_airline)
    fig_airline = _big_titles(fig_airline, size=24)  # << tamaño del título
    fig_airline.update_layout(autosize=True, height=500, showlegend=False, margin=dict(l=10, r=10, t=50, b=20))
    st.plotly_chart(fig_airline, use_container_width=True)

with logos_col:
    st.markdown("#### Detalle")
    total_vuelos_air = int(by_airline["VUELOS"].sum())
    assets_dir = _assets_dir(Path(__file__).resolve())

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
                    unsafe_allow_html=True
                )
        with rc:
            st.markdown(f"<span style='color:{color_txt};font-weight:700'>{name}</span>", unsafe_allow_html=True)
            st.markdown(f"{pct:.1f}% — {vuelos:,} vuelos")
        st.markdown("---")

st.markdown("---")

# ---------- Gráfico 3: Despegues vs Aterrizajes (con íconos) ----------
ICON_DESPEGUE   = Path(__file__).resolve().parents[1] / "assets" / "icons" / "plane_despegue.png"
ICON_ATERRIZAJE = Path(__file__).resolve().parents[1] / "assets" / "icons" / "plane_aterrizaje.png"

by_tipo_anio = (
    df.groupby(["AÑO", "TIPO_MOV"])
      .size()
      .reset_index(name="VUELOS")
)

fig_tipo = px.bar(
    by_tipo_anio,
    x="AÑO",
    y="VUELOS",
    color="TIPO_MOV",
    barmode="group",
    title="🛫Despegues vs 🛬Aterrizajes por año",
    color_discrete_map={"DESPEGUE": "#0077cc", "ATERRIZAJE": "#5bc0de"},
    category_orders={"TIPO_MOV": ["DESPEGUE", "ATERRIZAJE"]}
)
fig_tipo = style_fig(fig_tipo)
fig_tipo = _force_axes_colors(fig_tipo)
fig_tipo = _big_titles(fig_tipo, size=24)         # << tamaño del título

# --- añadir íconos sobre cada barra ---
try:
    img_despegue = Image.open(ICON_DESPEGUE)
    img_aterrizaje = Image.open(ICON_ATERRIZAJE)
    x_offset = 0.15  # separación horizontal

    for _, row in by_tipo_anio.iterrows():
        año = row["AÑO"]; tipo = row["TIPO_MOV"]; vuelos = float(row["VUELOS"])
        x_shift = -x_offset if tipo == "DESPEGUE" else x_offset
        img_src = img_despegue if tipo == "DESPEGUE" else img_aterrizaje
        fig_tipo.add_layout_image(dict(
            source=img_src, xref="x", yref="y",
            x=año + x_shift, y=vuelos / 2.5,
            sizex=1, sizey=vuelos * 0.4,
            xanchor="center", yanchor="middle", layer="above"
        ))
except Exception as e:
    st.warning(f"No se pudieron agregar los íconos de avión: {e}")

st.plotly_chart(fig_tipo, use_container_width=True)

st.markdown("---")
# --- Cargar detalle de aeropuertos (CSV con ;)
det_path = Path(__file__).resolve().parents[1] / "data" / "aeropuertos_detalle.csv"
det = pd.read_csv(
    det_path,
    sep=";",
    dtype=str,
    usecols=["local", "oaci", "iata", "denominacion"],
    encoding="utf-8"
).fillna("")

# Normalizo mayúsculas y espacios
for c in ["local", "oaci", "iata", "denominacion"]:
    det[c] = det[c].astype(str).str.strip()

det["local"] = det["local"].str.upper()
det["oaci"]  = det["oaci"].str.upper()
det["iata"]  = det["iata"].str.upper()

# Creo un mapping robusto: code -> denominacion (combina local/iata/oaci)
m = pd.melt(
    det,
    id_vars=["denominacion"],
    value_vars=["local", "iata", "oaci"],
    var_name="tipo",
    value_name="code"
)
m = m[m["code"] != ""]
m["code"] = m["code"].str.upper()
code_to_name = m.drop_duplicates("code").set_index("code")["denominacion"]

# === Top 10 aeropuertos más visitados (por código en df["AEROPUERTO"])
top10 = (
    df.groupby("AEROPUERTO")
      .size()
      .reset_index(name="VUELOS")
      .sort_values("VUELOS", ascending=False)
      .head(10)
)

# Mapear código -> nombre
top10["NOMBRE"] = top10["AEROPUERTO"].str.upper().map(code_to_name).fillna(top10["AEROPUERTO"])

# Gráfico (muestro el nombre completo en el eje Y)
fig_top = px.bar(
    top10.sort_values("VUELOS"),
    x="VUELOS",
    y="NOMBRE",
    orientation="h",
    title="🧭 Aeropuertos más transitados (Top 10)",
    text_auto=True
)
fig_top.update_traces(marker_color="#90d1e3")
fig_top.update_layout(xaxis_title="Cantidad de vuelos", yaxis_title="Aeropuerto")
fig_top = style_fig(fig_top)  # si usás tu helper de tema
st.plotly_chart(fig_top, use_container_width=True)

# Leyenda lateral código → nombre (opcional)
with st.expander("Ver códigos ↔ nombres"):
    for _, r in top10.sort_values("VUELOS", ascending=False).iterrows():
        st.markdown(f"- **{r['AEROPUERTO']}** → {r['NOMBRE']}")

st.markdown("---")

# ---------- Tabla de detalle ----------
st.markdown("### 📄 Detalle de vuelos (Muestra de 500 filas).")
st.dataframe(
    df[["Fecha", "AEROPUERTO", "AEROLINEA", "TIPO_MOV", "PAX", "AÑO", "MES"]].head(500),
    use_container_width=True
)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV filtrado", data=csv, file_name="vuelos_filtrado.csv", mime="text/csv")

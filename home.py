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

# === Import del tema global (ui_theme.py en la raíz) ===
try:
    from ui_theme import init_theme, style_fig, map_style
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from ui_theme import init_theme, style_fig, map_style

init_theme(show_toggle=False)  # modo oscuro fijo

from ui_theme import show_global_title
show_global_title()

# ========== SOLO LO NECESARIO PARA COLORES Y TABLA ==========
def _theme_vals():
    return st.session_state.get(
        "_theme",
        {"bg": "#0e1117", "text": "#e5e7eb", "accent": "#60a5fa"}
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
        yaxis=dict(title_font_color=t["text"], tickfont=dict(color=t["text"]))
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
        div[data-testid="stDataFrame"] * {{ color: {t['text']} !important; }}
        div[data-testid="stDataFrame"] thead tr th {{
            background: rgba(255,255,255,0.04) !important;
            font-weight: 600 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

_patch_light_css()

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
    if s is None: return ""
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
        p = assets_dir / "flybondi.png";  return p if p.exists() else None
    if "JETSMART" in n:
        p = assets_dir / "jetsmart.png";  return p if p.exists() else None
    if "AEROLINEAS" in n and "ARGENTINAS" in n:
        p = assets_dir / "aerolineas_argentinas.png";  return p if p.exists() else None
    return None

def _icons_dir(this_file: Path) -> Path:
    candidates = [
        this_file.parent / "assets" / "icons",          # si el file está en raíz
        this_file.parent.parent / "assets" / "icons"    # si el file está en /pages
    ]
    for c in candidates:
        if c.exists():
            return c
    return this_file.parent / "assets" / "icons"

# === Rutas robustas y carga de datos ===
DATA_DIR = _find_data_dir(Path(__file__).resolve())

@st.cache_data(show_spinner=True)
def load_data():
    try:
        # SOLO parseamos "Fecha" (seguro en tus CSV)
        eda = pd.read_csv(DATA_DIR / "vuelos_eda.csv", low_memory=False, parse_dates=["Fecha"])
        aero = pd.read_csv(DATA_DIR / "aeropuertos_eda.csv", low_memory=False)

        eda["AEROLINEA"] = (_std_text(eda["Aerolinea Nombre"])
                            if "Aerolinea Nombre" in eda.columns
                            else _std_text(eda.get("Aerolinea", pd.Series(index=eda.index))))
        eda["AEROPUERTO"] = _std_text(eda.get("Aeropuerto", pd.Series(index=eda.index)))
        eda["TIPO_MOV"]   = _std_text(eda.get("Tipo Movimiento", pd.Series(index=eda.index)))
        eda["PAX"]        = pd.to_numeric(eda.get("Pasajeros", np.nan), errors="coerce")
        eda["AÑO"]        = eda["Fecha"].dt.year
        eda["MES"]        = eda["Fecha"].dt.to_period("M").dt.to_timestamp()

        if "denominacion" in aero.columns:
            aero["DENOMINACION_STD"] = _std_text(aero["denominacion"])
        else:
            aero["DENOMINACION_STD"] = ""
        return eda, aero
    except FileNotFoundError as e:
        st.error(f"❌ No encontré **{e.filename}** en {DATA_DIR}")
        raise
    except Exception as e:
        st.error(f"❌ Error cargando datos desde {DATA_DIR}: {e}")
        raise

eda, aero = load_data()

# === Soporte nombres de aeropuertos (labels bonitas) ===
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
    m = pd.melt(det, id_vars=["denominacion"],
                value_vars=[c for c in ["local","iata","oaci"] if c in det.columns],
                var_name="tipo", value_name="code")
    m = m[m["code"].ne("")]
    return m.drop_duplicates("code").set_index("code")["denominacion"].to_dict()

code_to_name = _airport_code_to_name(DATA_DIR)

# ---------- Sidebar (Filtros con INCLUIR / EXCLUIR) ----------
st.sidebar.header("Filtros")

def _apply(df, col, values, exclude=False):
    if not values:
        return df
    return df[~df[col].isin(values)] if exclude else df[df[col].isin(values)]

# Listas base
years = sorted(eda["AÑO"].dropna().unique().tolist())
aerolineas = sorted(eda["AEROLINEA"].dropna().unique().tolist())
tipos = sorted(eda["TIPO_MOV"].dropna().unique().tolist())
all_codes = sorted(eda["AEROPUERTO"].dropna().unique().tolist())
airport_labels = [f"{c} — {code_to_name.get(c, c)}" for c in all_codes]
code_from_label = {lab: lab.split(" — ", 1)[0] for lab in airport_labels}

# Año
with st.sidebar.container():
    st.caption("Año")
    sel_years = st.multiselect("Seleccioná años", years, key="year_ms")
    excl_years = st.checkbox("Excluir selección", key="year_excl", value=False)

# Mes
with st.sidebar.container():
    st.caption("Mes")
    _meses = {
        1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio",
        7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"
    }
    month_labels = [_meses[m] for m in range(1,13)]
    label_to_num = {v:k for k,v in _meses.items()}

    sel_month_labels = st.multiselect("Seleccioná meses", options=month_labels, key="month_ms")
    sel_months = [label_to_num[l] for l in sel_month_labels]
    excl_month = st.checkbox("Excluir selección", key="month_excl", value=False)

# Aerolínea
with st.sidebar.container():
    st.caption("Aerolínea")
    sel_aerolinea = st.multiselect("Seleccioná aerolíneas", aerolineas, key="airline_ms")
    excl_airline = st.checkbox("Excluir selección", key="airline_excl", value=False)

# Aeropuerto (con label bonito)
with st.sidebar.container():
    st.caption("Aeropuerto")
    sel_airport_labels = st.multiselect("Seleccioná aeropuertos", options=airport_labels, key="airport_ms")
    sel_airports = [code_from_label[s] for s in sel_airport_labels]
    excl_airport = st.checkbox("Excluir selección", key="airport_excl", value=False)

# Tipo de movimiento
with st.sidebar.container():
    st.caption("Tipo de movimiento")
    sel_tipo = st.multiselect("Seleccioná tipos", tipos, key="tipo_ms")
    excl_tipo = st.checkbox("Excluir selección", key="tipo_excl", value=False)


# Aplicación de filtros
df = eda.copy()
df = _apply(df, "AÑO", sel_years, exclude=excl_years)

# ⬇️ Mes por Fecha (usa dt.month)
if sel_months:
    df = df[~df["Fecha"].dt.month.isin(sel_months)] if excl_month \
         else df[df["Fecha"].dt.month.isin(sel_months)]

df = _apply(df, "AEROLINEA", sel_aerolinea, exclude=excl_airline)
df = _apply(df, "TIPO_MOV", sel_tipo, exclude=excl_tipo)
df = _apply(df, "AEROPUERTO", sel_airports, exclude=excl_airport)


# ================= Contenido principal =================

# ---------- KPIs ----------
st.markdown("""
<style>
div[data-testid="stMetricLabel"] { display: none !important; }
.kpi-title{ font-size:28px; font-weight:400; color:#e5e7eb; line-height:1.15; margin:0 0 -6px 0;}
div[data-testid="stMetricValue"]{ font-size:34px !important; font-weight:800 !important; color:#ffffff !important;}
</style>
""", unsafe_allow_html=True)

total_vuelos = len(df)
total_pax = int(df["PAX"].sum()) if df["PAX"].notna().any() else None
aerolineas_activas = int(df["AEROLINEA"].nunique())
aeropuertos_activos = int(df["AEROPUERTO"].nunique())

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown('<div class="kpi-title">✈️ Vuelos</div>', unsafe_allow_html=True); st.metric("", f"{total_vuelos:,}")
with c2: st.markdown('<div class="kpi-title">🧍 PAX</div>', unsafe_allow_html=True);  st.metric("", (f"{total_pax:,}" if total_pax is not None else "—"))
with c3: st.markdown('<div class="kpi-title">🏢 Aerolíneas</div>', unsafe_allow_html=True); st.metric("", f"{aerolineas_activas:,}")
with c4: st.markdown('<div class="kpi-title">🛫 Aeropuertos</div>', unsafe_allow_html=True); st.metric("", f"{aeropuertos_activos:,}")
st.markdown("---")

# ---------- Gráfico 1: Barras por año ----------
ICONS_DIR = _icons_dir(Path(__file__).resolve())
ICON_PATH = ICONS_DIR / "plane.png"
by_year = df.groupby("AÑO").size().reset_index(name="VUELOS").sort_values("VUELOS", ascending=True)

fig_year = px.bar(by_year, x="VUELOS", y="AÑO", orientation="h", title="✈️ Cantidad de vuelos por año", text_auto=True)
fig_year.update_traces(marker_color="#7ec8e3")
fig_year.update_layout(xaxis_title="Vuelos", yaxis_title="Año", yaxis=dict(categoryorder="total ascending"))

if ICON_PATH.exists():
    try:
        plane_img = Image.open(ICON_PATH)
        x_max = float(by_year["VUELOS"].max()); pad = 0.2 * x_max
        fig_year.update_xaxes(range=[0, x_max + pad])
        size_x, size_y = 2 * x_max, 2
        for _, r in by_year.iterrows():
            fig_year.add_layout_image(dict(
                source=plane_img, xref="x", yref="y", x=float(r["VUELOS"]), y=r["AÑO"],
                sizex=size_x, sizey=size_y, xanchor="left", yanchor="middle", layer="above"
            ))
    except Exception as e:
        st.warning(f"No se pudo cargar el ícono del avión: {e}")
else:
    st.warning("No se encontró el ícono del avión en assets/icons/plane.png")

fig_year = style_fig(fig_year); fig_year = _force_axes_colors(fig_year); fig_year = _big_titles(fig_year, size=24)
st.plotly_chart(fig_year, use_container_width=True)
st.markdown("---")

# ---------- Gráfico: Vuelos por mes (acumulado multi-año con estética aérea mejorada) ----------
st.markdown("### 🗓️✈️ Vuelos por mes (acumulado multi-año)")

if not df.empty:
    tmp = df.copy()
    tmp["_MES_NUM"] = tmp["Fecha"].dt.month  # 1..12

    mes_nombres = {
        1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio",
        7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"
    }

    by_month_cal = (
        tmp.groupby("_MES_NUM")
           .size()
           .reindex(range(1, 12+1), fill_value=0)
           .reset_index(name="VUELOS")
           .rename(columns={"_MES_NUM": "MES_NUM"})
    )
    by_month_cal["MES"] = by_month_cal["MES_NUM"].map(mes_nombres)
    by_month_cal["MES"] = pd.Categorical(
        by_month_cal["MES"],
        categories=[mes_nombres[i] for i in range(1, 13)],
        ordered=True
    )
    total_all = int(by_month_cal["VUELOS"].sum())
    by_month_cal["PCT"] = (by_month_cal["VUELOS"] / total_all).fillna(0)

    highlight_months = {1, 7, 8}
    bar_colors = ["#7ec8e3" if m in highlight_months else "#60a5fa" for m in by_month_cal["MES_NUM"]]

    fig_month_tot = px.bar(
        by_month_cal.sort_values("MES"),
        x="MES", y="VUELOS",
        text=by_month_cal["VUELOS"].map(lambda x: f"{x:,}")
    )

    fig_month_tot.update_traces(
        marker_color=bar_colors,
        # ✅ dentro de las barras
        insidetextanchor="middle",
        textfont=dict(size=16, color="#0e1117", weight="bold"),  # ✅ fuente más grande, legible sobre azul, texto en negrita
        cliponaxis=False,
        hovertemplate="🗓️ <b>%{x}</b><br>✈️ Vuelos: %{y:,}<br>% de participación: %{customdata:.1%}<extra></extra>",
        customdata=by_month_cal["PCT"].to_numpy()
    )

    ymax = float(by_month_cal["VUELOS"].max())
    fig_month_tot.add_shape(
        type="rect", xref="x", yref="y",
        x0=by_month_cal["MES"].iloc[0], x1=by_month_cal["MES"].iloc[-1],
        y0=-ymax*0.02, y1=0,
        line=dict(width=0), fillcolor="rgba(255,255,255,0.08)"
    )

    fig_month_tot.update_layout(barcornerradius=6)

    # ✈️ Avioncitos más grandes
    try:
        ICONS_DIR = _icons_dir(Path(__file__).resolve())
        plane_icon = Image.open(ICONS_DIR / "plane.png")
        for _, r in by_month_cal.iterrows():
            fig_month_tot.add_layout_image(dict(
                source=plane_icon,
                xref="x", yref="y",
                x=r["MES"], y=r["VUELOS"] * 0.5,  # apenas arriba
                sizex=1.2, sizey=ymax * 0.25,      # ✅ mucho más grandes
                xanchor="center", yanchor="bottom", layer="above"
            ))
    except Exception:
        pass

    fig_month_tot.update_layout(
        xaxis_title="Mes",
        yaxis_title="Cantidad de vuelos",
        xaxis=dict(tickangle=0, showgrid=False),
        margin=dict(l=10, r=10, t=60, b=40)
    )

    fig_month_tot = style_fig(fig_month_tot)
    fig_month_tot = _force_axes_colors(fig_month_tot)
    fig_month_tot = _big_titles(fig_month_tot, size=24)
    st.plotly_chart(fig_month_tot, use_container_width=True)
else:
    st.info("No hay datos para los filtros seleccionados.")

st.markdown("---")

# ---------- Gráfico 2: Donut por aerolínea + logos ----------
TOP_N = 10
by_airline = (df.groupby("AEROLINEA").size().reset_index(name="VUELOS").sort_values("VUELOS", ascending=False).head(TOP_N))
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
    fig_airline = px.pie(by_airline, names="AEROLINEA", values="VUELOS",
                         color="AEROLINEA_NORM", color_discrete_map=custom_colors_norm,
                         title="Participación por Aerolínea", hole=0.35)
    fig_airline.update_traces(textinfo="percent+label", textfont_size=16, insidetextorientation="radial",
                              pull=[0.05]*len(by_airline), marker=dict(line=dict(width=0)))
    fig_airline = style_fig(fig_airline); fig_airline = _force_axes_colors(fig_airline); fig_airline = _big_titles(fig_airline, size=24)
    fig_airline.update_layout(autosize=True, height=500, showlegend=False, margin=dict(l=10, r=10, t=50, b=20))
    st.plotly_chart(fig_airline, use_container_width=True)

assets_dir = _assets_dir(Path(__file__).resolve())
with logos_col:
    st.markdown("#### Detalle")
    total_vuelos_air = int(by_airline["VUELOS"].sum())
    for _, row in by_airline.iterrows():
        name = str(row["AEROLINEA"]); name_norm = _normalize_name(name)
        vuelos = int(row["VUELOS"]);  pct = 100.0 * vuelos / total_vuelos_air if total_vuelos_air > 0 else 0.0
        color_txt = custom_colors_norm.get(name_norm, default_color)
        lc, rc = st.columns([1, 2.4])
        with lc:
            logo_path = _logo_path_for_airline(name, assets_dir)
            if logo_path: st.image(str(logo_path), use_container_width=True)
            else:
                st.markdown("<div style='width:100%;height:52px;border:1px dashed #555;display:flex;align-items:center;justify-content:center;'><span style='font-size:11px;color:#aaa;'>Sin logo</span></div>", unsafe_allow_html=True)
        with rc:
            st.markdown(f"<span style='color:{color_txt};font-weight:700'>{name}</span>", unsafe_allow_html=True)
            st.markdown(f"{pct:.1f}% — {vuelos:,} vuelos")
        st.markdown("---")

st.markdown("---")

# ---------- Gráfico 3: Despegues vs Aterrizajes ----------
ICON_DESPEGUE   = ICONS_DIR / "plane_despegue.png"
ICON_ATERRIZAJE = ICONS_DIR / "plane_aterrizaje.png"

by_tipo_anio = df.groupby(["AÑO", "TIPO_MOV"]).size().reset_index(name="VUELOS")
fig_tipo = px.bar(by_tipo_anio, x="AÑO", y="VUELOS", color="TIPO_MOV", barmode="group",
                  title="🛫Despegues vs 🛬Aterrizajes por año",
                  color_discrete_map={"DESPEGUE": "#0077cc", "ATERRIZAJE": "#5bc0de"},
                  category_orders={"TIPO_MOV": ["DESPEGUE", "ATERRIZAJE"]})
fig_tipo = style_fig(fig_tipo); fig_tipo = _force_axes_colors(fig_tipo); fig_tipo = _big_titles(fig_tipo, size=24)

try:
    img_despegue = Image.open(ICON_DESPEGUE)
    img_aterrizaje = Image.open(ICON_ATERRIZAJE)
    x_offset = 0.15
    for _, row in by_tipo_anio.iterrows():
        año = row["AÑO"]; tipo = row["TIPO_MOV"]; vuelos = float(row["VUELOS"])
        x_shift = -x_offset if tipo == "DESPEGUE" else x_offset
        img_src = img_despegue if tipo == "DESPEGUE" else img_aterrizaje
        fig_tipo.add_layout_image(dict(source=img_src, xref="x", yref="y",
                                       x=año + x_shift, y=vuelos / 2.5,
                                       sizex=1, sizey=vuelos * 0.4,
                                       xanchor="center", yanchor="middle", layer="above"))
except Exception as e:
    st.warning(f"No se pudieron agregar los íconos de avión: {e}")

st.plotly_chart(fig_tipo, use_container_width=True)
st.markdown("---")

# --- Detalle de aeropuertos (usa DATA_DIR) ---
det_path = DATA_DIR / "aeropuertos_detalle.csv"
if det_path.exists():
    det = pd.read_csv(det_path, sep=";", dtype=str,
                      usecols=["local", "oaci", "iata", "denominacion"],
                      encoding="utf-8").fillna("")
    for c in ["local", "oaci", "iata", "denominacion"]:
        det[c] = det[c].astype(str).str.strip()
    det["local"] = det["local"].str.upper()
    det["oaci"]  = det["oaci"].str.upper()
    det["iata"]  = det["iata"].str.upper()

    m = pd.melt(det, id_vars=["denominacion"], value_vars=["local", "iata", "oaci"],
                var_name="tipo", value_name="code")
    m = m[m["code"] != ""]
    m["code"] = m["code"].str.upper()
    code_to_name = m.drop_duplicates("code").set_index("code")["denominacion"]

    top10 = (df.groupby("AEROPUERTO").size().reset_index(name="VUELOS")
               .sort_values("VUELOS", ascending=False).head(10))
    top10["NOMBRE"] = top10["AEROPUERTO"].str.upper().map(code_to_name).fillna(top10["AEROPUERTO"])


    st.markdown("#### 🧭 Aeropuertos más activos")
    fig_top = px.bar(top10.sort_values("VUELOS"), x="VUELOS", y="NOMBRE", orientation="h",
                     title=" ", text_auto=True)
    fig_top.update_traces(marker_color="#90d1e3")
    fig_top.update_layout(xaxis_title="Cantidad de vuelos", yaxis_title="Aeropuerto")
    fig_top = style_fig(fig_top)
    st.plotly_chart(fig_top, use_container_width=True)

else:
    st.warning(f"No encontré {det_path.name} en {DATA_DIR}")

st.markdown("---")

# ---------- Widget: Top conexiones Origen → Destino ----------
# Tus columnas son fijas en este dataset:
ORIG_COL = "Aeropuerto"
DEST_COL = "Origen / Destino"

# --- Construcción del DF de rutas ---
TOP_N_RUTAS = 10

rutas = (
    df[[ORIG_COL, DEST_COL]]
    .dropna()
    .astype(str)
    .assign(
        **{
            ORIG_COL: lambda d: d[ORIG_COL].str.strip().str.upper(),
            DEST_COL: lambda d: d[DEST_COL].str.strip().str.upper()
        }
    )
    .value_counts(sort=True)
    .reset_index(name="VUELOS")
    .head(TOP_N_RUTAS)
)
# ---------- Widget: Top conexiones Origen → Destino (con filtro por origen/destino y sin línea punteada) ----------
import plotly.graph_objects as go
import textwrap

st.markdown("### 🔀✈️ Top conexiones Origen → Destino")

ORIG_COL_RAW = "Aeropuerto"          # en el CSV: origen
DEST_COL_RAW = "Origen / Destino"    # en el CSV: destino
TOP_N_RUTAS = 10

# === Dataset para este widget: aplica TODOS los filtros excepto el de aeropuerto,
#     que aquí se trata como (Origen O Destino) y también para excluir. ===
df_routes = eda.copy()

# Año
df_routes = _apply(df_routes, "AÑO", sel_years, exclude=excl_years)

# Mes (por Fecha.dt.month)
if sel_months:
    df_routes = df_routes[~df_routes["Fecha"].dt.month.isin(sel_months)] if excl_month \
                else df_routes[df_routes["Fecha"].dt.month.isin(sel_months)]

# Aerolínea y tipo de movimiento
df_routes = _apply(df_routes, "AEROLINEA", sel_aerolinea, exclude=excl_airline)
df_routes = _apply(df_routes, "TIPO_MOV", sel_tipo,       exclude=excl_tipo)

# Aeropuerto (especial: considerar ORIGEN O DESTINO)
if sel_airports:
    if excl_airport:
        df_routes = df_routes[~(
            df_routes[ORIG_COL_RAW].isin(sel_airports) |
            df_routes[DEST_COL_RAW].isin(sel_airports)
        )]
    else:
        df_routes = df_routes[
            df_routes[ORIG_COL_RAW].isin(sel_airports) |
            df_routes[DEST_COL_RAW].isin(sel_airports)
        ]

# === Agregado de rutas (Vuelos y PAX) ===
rutas = (
    df_routes.groupby([ORIG_COL_RAW, DEST_COL_RAW])
             .agg(VUELOS=('Fecha', 'size'), PAX=('PAX', 'sum'))
             .reset_index()
             .sort_values('VUELOS', ascending=False)
             .head(TOP_N_RUTAS)
             .astype({ORIG_COL_RAW: str, DEST_COL_RAW: str})
)

def _pretty_with_code(code: str) -> str:
    su = str(code or "").upper().strip()
    full = code_to_name.get(su)
    return f"{full} ({su})" if full else su

def _wrap_html(txt: str, width: int = 28) -> str:
    try:
        return "<br>".join(textwrap.wrap(str(txt), width=width))
    except Exception:
        return str(txt)

rutas["ORIGEN_NICE"]  = rutas[ORIG_COL_RAW].apply(_pretty_with_code)
rutas["DESTINO_NICE"] = rutas[DEST_COL_RAW].apply(_pretty_with_code)

if rutas.empty:
    st.info("No hay rutas para los filtros actuales.")
else:
    n = len(rutas)
    fig_routes = go.Figure()

    # Canvas
    fig_routes.update_xaxes(visible=False, range=[0, 1])
    fig_routes.update_yaxes(visible=False, range=[0, n + 1.8])

    # Colores y bandas
    base_color   = "#60a5fa"
    accent_color = "#e5e7eb"
    band_a       = "rgba(255,255,255,0.03)"
    band_b       = "rgba(255,255,255,0.00)"

    # Columnas (posiciones X)
    X_NUM        = 0.05
    X_ORIG       = 0.18
    X_STATS      = 0.36
    X_ARROW_A    = 0.46
    X_ARROW_B    = 0.74
    DEST_PANEL_L = 0.80   # panel de destino (sin línea punteada)
    DEST_PANEL_R = 0.97
    X_DEST       = 0.88
    X_PLANE      = 0.60
    Y_PLANE_OFF  = 0.30

    # Panel de destino (SIN línea punteada)
    fig_routes.add_shape(
        type="rect", xref="x", yref="y",
        x0=DEST_PANEL_L, x1=DEST_PANEL_R, y0=0.2, y1=n + 1.2,
        line=dict(width=0)
    )

    # Bandas por fila + sutil separador
    for rank in range(1, n + 1):
        y = n - rank + 1
        color = band_a if rank % 2 else band_b
        fig_routes.add_shape(type="rect", xref="x", yref="y",
                             x0=0.10, x1=DEST_PANEL_R, y0=y - 0.48, y1=y + 0.48,
                             line=dict(width=0), fillcolor=color)
        fig_routes.add_shape(type="line", xref="x", yref="y",
                             x0=0.10, x1=DEST_PANEL_R, y0=y - 0.5, y1=y - 0.5,
                             line=dict(color="rgba(255,255,255,0.06)", width=1))

    # Filas
    for rank, r in enumerate(rutas.itertuples(index=False), start=1):
        y = n - rank + 1
        origen  = _wrap_html(r.ORIGEN_NICE,  width=30)
        destino = _wrap_html(r.DESTINO_NICE, width=26)
        vuelos  = int(r.VUELOS)
        pax     = int(0 if pd.isna(r.PAX) else r.PAX)

        # Número de ranking
        fig_routes.add_annotation(x=X_NUM, y=y, xref="x", yref="y",
                                  text=f"<b style='font-size:22px'>{rank}</b>",
                                  showarrow=False, font=dict(size=22, color=accent_color))

        # Origen
        fig_routes.add_annotation(x=X_ORIG, y=y, xref="x", yref="y",
                                  text=origen, showarrow=False, align="left",
                                  font=dict(size=16, color=accent_color))

        # Stats (PAX/Vuelos) en columna intermedia
        fig_routes.add_annotation(x=X_STATS, y=y, xref="x", yref="y",
                                  text=f"👥 PAX: <b>{pax:,}</b><br>✈️ Vuelos: <b>{vuelos:,}</b>",
                                  showarrow=False, align="center",
                                  font=dict(size=14, color=base_color))

        # Flecha más corta/angosta
        fig_routes.add_annotation(x=X_ARROW_B, y=y, ax=X_ARROW_A, ay=y,
                                  xref="x", yref="y", axref="x", ayref="y",
                                  showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3,
                                  arrowcolor=base_color)

        # Destino (en panel)
        fig_routes.add_annotation(x=X_DEST, y=y, xref="x", yref="y",
                                  text=destino, showarrow=False, align="center",
                                  font=dict(size=16, color=accent_color))

    # Avión
    try:
        ICONS_DIR = _icons_dir(Path(__file__).resolve())
        plane_icon = Image.open(ICONS_DIR / "plane.png")
        for rank in range(1, n + 1):
            y = n - rank + 1
            fig_routes.add_layout_image(dict(
                source=plane_icon, xref="x", yref="y",
                x=X_PLANE, y=y + Y_PLANE_OFF,
                sizex=0.18, sizey=0.90,
                xanchor="center", yanchor="middle", layer="above"
            ))
    except Exception:
        pass

    # Layout
    fig_routes.update_layout(
        margin=dict(l=40, r=40, t=60, b=28),
        height=max(440, 67 * n),
        title=""
    )
    fig_routes = style_fig(fig_routes)
    fig_routes = _force_axes_colors(fig_routes)
    fig_routes = _big_titles(fig_routes, size=24)
    st.plotly_chart(fig_routes, use_container_width=True)

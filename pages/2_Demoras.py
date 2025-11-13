# -*- coding: utf-8 -*-
# pages/2_Demoras.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import base64

st.set_page_config(page_title="Demoras", layout="wide")

# ======================================================
# UTILS
# ======================================================

def _theme_vals():
    return st.session_state.get(
        "_theme",
        {"bg": "#0e1117", "text": "#e5e7eb", "accent": "#60a5fa"}
    )

def _force_axes_colors(fig):
    t = _theme_vals()
    fig.update_layout(
        font=dict(color=t["text"]),
        title_font_color=t["text"],
        xaxis=dict(title_font_color=t["text"], tickfont=dict(color=t["text"])),
        yaxis=dict(title_font_color=t["text"], tickfont=dict(color=t["text"]))
    )
    return fig

def _big_titles(fig, size=24):
    fig.update_layout(title=dict(font=dict(size=size)))
    return fig


# ======================================================
# SIDEBAR HEADER (UADE)
# ======================================================

def _uade_sidebar_header():
    logo_path = Path("assets/logos/uade.png")
    if not logo_path.exists():
        return

    img_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebarNav"] {{
            background-image: url("data:image/png;base64,{img_b64}");
            background-repeat: no-repeat;
            background-position: 16px 8px;
            background-size: 120px auto;
            padding-top: 75px;
        }}
        [data-testid="stSidebarNav"]::before {{
            white-space: pre-line;
            display: block;
            margin: 0 0 60px 0;
            padding-left: 10px;
            border-left: 1px solid rgba(255,255,255,0.35);
            font-size: 12px;
            color: #e5e7eb;
            content: "Grupo 1 - Ciencia de Datos\\AReinaldo Barreto\\AOriana Bianchi\\A Federico Rey\\AGabriel Kraus";
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

_uade_sidebar_header()

from ui_theme import init_theme, style_fig, show_global_title
init_theme(show_toggle=False)
show_global_title()


# ======================================================
# DESCRIPCIÓN
# ======================================================

st.markdown("""
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
    Análisis de demoras reales comparando tiempo estimado vs tiempo ejecutado,
    con filtros avanzados por aeropuerto, ruta, mes, año y aerolínea.
</div>
""", unsafe_allow_html=True)


# ======================================================
# CARGA DE DATOS
# ======================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

@st.cache_data
def load_vuelos(data_dir):
    df = pd.read_csv(
        data_dir / "vuelos_preliminar.csv",
        low_memory=False,
        parse_dates=["Fecha","Fecha_Hora_Local_despegue","Fecha_Hora_Local_aterrizaje"]
    )
    df["AÑO"] = df["Fecha"].dt.year
    df["MES"] = df["Fecha"].dt.month

    def to_min(t):
        try:
            h,m,s = map(int, str(t).split(":"))
            return h*60 + m + s/60
        except:
            return None

    df["REAL_MIN"] = df["Tiempo_Vuelo"].apply(to_min)
    return df

@st.cache_data
def load_promedios(data_dir):
    df = pd.read_csv(data_dir / "promedio_tiempos_vuelo.csv")
    df["Tiempo_Promedio_MIN"] = df["Tiempo_Promedio"].apply(
        lambda t: sum(int(x)*60**i for i,x in enumerate(reversed(t.split(":"))))/60
    )
    return df

df_vuelos = load_vuelos(DATA_DIR)
df_prom = load_promedios(DATA_DIR)

df = df_vuelos.merge(df_prom, on=["Aerolinea","Origen","Destino"], how="left")
df["DEMORA_MIN"] = df["REAL_MIN"] - df["Tiempo_Promedio_MIN"]
df["DEMORADO_15"] = df["DEMORA_MIN"] > 15

def classify_delay(m):
    if pd.isna(m): return "Sin dato"
    if m < 15: return "Sin demora (<15)"
    if m < 30: return "15–30 min"
    if m < 60: return "30–60 min"
    return "60+ min"

df["CATEGORIA_DEMORA"] = df["DEMORA_MIN"].apply(classify_delay)


# ======================================================
# AEROPUERTOS DETALLE — MAPEOS (versión completa COMO EN General.py)
# ======================================================

def load_airport_mapping(data_dir):
    p = data_dir / "aeropuertos_detalle.csv"
    det = pd.read_csv(p, sep=";", dtype=str).fillna("")

    # Normalizar
    for c in ["local", "oaci", "iata", "denominacion"]:
        if c in det.columns:
            det[c] = det[c].astype(str).str.strip().str.upper()

    # Unir local + oaci + iata para mapear todo
    melt = pd.melt(
        det,
        id_vars=["denominacion"],
        value_vars=["local", "oaci", "iata"],
        var_name="tipo",
        value_name="code",
    )

    melt = melt[melt["code"] != ""]

    # Diccionario final: cualquier código → denominación
    return melt.drop_duplicates("code").set_index("code")["denominacion"].to_dict()


# Cargar mapeo completo
CODE_TO_NAME = load_airport_mapping(DATA_DIR)

# Crear etiquetas “bonitas”
df["ORIGEN_NOMBRE"] = df["Origen"].apply(lambda x: f"{x} – {CODE_TO_NAME.get(x, 'SIN NOMBRE')}")
df["DESTINO_NOMBRE"] = df["Destino"].apply(lambda x: f"{x} – {CODE_TO_NAME.get(x, 'SIN NOMBRE')}")

# ======================================================
# 🎛️ FILTROS — usando nombres largos
# ======================================================

st.markdown("### 🎛️ Filtros")

colA, colB, colC, colD = st.columns(4)

years = sorted(df["AÑO"].unique())
airlines = sorted(df["Aerolinea"].unique())
meses = {
    1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
    7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
}

# AÑO
with colA:
    sel_years = st.multiselect("Año", years)

# MES
with colB:
    sel_months_labels = st.multiselect("Mes", list(meses.values()))
    sel_months = [k for k,v in meses.items() if v in sel_months_labels]

# AEROLÍNEA
with colC:
    sel_air = st.multiselect("Aerolínea", airlines)

# ORIGEN — nombre bonito
with colD:
    sel_origen_nice = st.multiselect(
        "Origen",
        options=sorted(df["ORIGEN_NOMBRE"].unique())
    )

# EXTRAER EL CÓDIGO REAL DE “AAA – NOMBRE…”
sel_origen = [
    x.split(" – ", 1)[0]  # Antes era NOMBRE_TO_IATA[x]
    for x in sel_origen_nice
] if sel_origen_nice else []

# DESTINO — nombre bonito
if sel_origen:
    opciones_dest = sorted(df[df["Origen"].isin(sel_origen)]["DESTINO_NOMBRE"].unique())
else:
    opciones_dest = sorted(df["DESTINO_NOMBRE"].unique())

sel_dest_nice = st.multiselect("Destino", opciones_dest)

sel_dest = [
    x.split(" – ", 1)[0]
    for x in sel_dest_nice
] if sel_dest_nice else []


# Aplicación de filtros
df_f = df.copy()
if sel_years: df_f = df_f[df_f["AÑO"].isin(sel_years)]
if sel_months: df_f = df_f[df_f["MES"].isin(sel_months)]
if sel_air: df_f = df_f[df_f["Aerolinea"].isin(sel_air)]
if sel_origen: df_f = df_f[df_f["Origen"].isin(sel_origen)]
if sel_dest: df_f = df_f[df_f["Destino"].isin(sel_dest)]

st.markdown("---")


# ======================================================
# KPIs (sin cambios)
# ======================================================

st.markdown("""
<style>
.kpi-title{ font-size:28px; font-weight:400; color:#e5e7eb; margin-bottom:-6px;}
div[data-testid="stMetricValue"]{ font-size:34px !important; font-weight:900 !important;}
</style>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)

total = df_f["Vuelo_ID"].nunique()
porc_dem = df_f["DEMORADO_15"].mean()*100 if total>0 else 0

ruta_top = df_f[df_f["DEMORADO_15"]].groupby(["Origen","Destino"]).size().sort_values(ascending=False)
ruta_top = f"{ruta_top.index[0][0]} → {ruta_top.index[0][1]}" if not ruta_top.empty else "—"

top_air = df_f[df_f["DEMORADO_15"]].groupby("Origen").size().sort_values(ascending=False)
air_top = top_air.index[0] if not top_air.empty else "—"

with c1: st.markdown('<div class="kpi-title">✈️ Vuelos analizados</div>', unsafe_allow_html=True); st.metric("", f"{total:,}")
with c2: st.markdown('<div class="kpi-title">🔥 % demoras (>15m)</div>', unsafe_allow_html=True); st.metric("", f"{porc_dem:.1f}%")
with c3: st.markdown('<div class="kpi-title">🛫 Aeropuerto con más demoras</div>', unsafe_allow_html=True); st.metric("", air_top)
with c4: st.markdown('<div class="kpi-title">📍 Ruta más afectada</div>', unsafe_allow_html=True); st.metric("", ruta_top)

st.markdown("---")

# ======================================================
# TABLA HTML PREMIUM — con pasajeros afectados
# ======================================================

st.markdown("## 🛫 Matriz de demoras por Aerolínea")

categoria_orden = ["Sin demora (<15)", "15–30 min", "30–60 min", "60+ min"]

tabla = (
    df_f.groupby(["Aerolinea","CATEGORIA_DEMORA"])
        .agg(
            CANTIDAD=("Vuelo_ID","nunique"),
            PAX=("Pasajeros","sum")
        )
        .reset_index()
)

tot = (
    df_f.groupby("Aerolinea")
        .agg(
            TOTAL_VUELOS=("Vuelo_ID","nunique"),
            TOTAL_PAX=("Pasajeros","sum")
        )
        .reset_index()
)

tabla = tabla.merge(tot,on="Aerolinea")
tabla["PCT"] = tabla["CANTIDAD"] / tabla["TOTAL_VUELOS"]

# Pivot inicial
pivot = tabla.pivot_table(
    index="Aerolinea",
    columns="CATEGORIA_DEMORA",
    values=["CANTIDAD","PAX","PCT"],
    fill_value=0
)
pivot.columns = [f"{a}_{b}" for a,b in pivot.columns]

# ======================================================
# 🔧 REINDEX ROBUSTO — evita error de columnas faltantes
# ======================================================

# Detectar qué categorías existen en la data filtrada
categorias_reales = sorted(df_f["CATEGORIA_DEMORA"].dropna().unique())

# Mantener solo categorías válidas que existan en orden esperado
orden_final = [c for c in categoria_orden if c in categorias_reales]

# Construir listas de columnas completas
cols_cant = [f"CANTIDAD_{c}" for c in orden_final]
cols_pax  = [f"PAX_{c}"      for c in orden_final]
cols_pct  = [f"PCT_{c}"      for c in orden_final]

# Forzar columnas aunque no existan
all_cols = cols_cant + cols_pax + cols_pct

# Crear las columnas faltantes con 0
for col in all_cols:
    if col not in pivot.columns:
        pivot[col] = 0

pivot = pivot[all_cols]      # reordenar
pivot = pivot.fillna(0)      # por si queda NaN

# Agregar totales
pivot["TOTAL"] = pivot[cols_cant].sum(axis=1)
pivot["TOTAL_PAX"] = pivot[cols_pax].sum(axis=1)

# Fila TOTAL global
fila_total = {}
total_v = df_f["Vuelo_ID"].nunique()
total_p = df_f["Pasajeros"].sum()

for c in orden_final:
    subset = df_f[df_f["CATEGORIA_DEMORA"] == c]
    fila_total[f"CANTIDAD_{c}"] = subset["Vuelo_ID"].nunique()
    fila_total[f"PAX_{c}"] = subset["Pasajeros"].sum()
    fila_total[f"PCT_{c}"] = (
        fila_total[f"CANTIDAD_{c}"]/total_v if total_v>0 else 0
    )

fila_total["TOTAL"] = total_v
fila_total["TOTAL_PAX"] = total_p

pivot.loc["TOTAL (TODAS)"] = fila_total

# ======================================================
# 🎨 HTML (sin cambios)
# ======================================================

def color_for_category(cat):
    if "Sin demora" in cat: return "#3CB371"
    if "15–30" in cat: return "#F7D358"
    if "30–60" in cat: return "#F39C12"
    return "#E74C3C"

html = """<style>
.demora-table{
    width:100%; border-collapse:separate; border-spacing:0 8px;
    font-size:18px; font-family:'Segoe UI', sans-serif; color:#e5e7eb;
}
.demora-table th{
    padding:12px; text-align:center; font-size:20px;
    background:rgba(255,255,255,0.06);
}
.demora-table td{
    padding:14px; text-align:center;
    background:rgba(255,255,255,0.04); border-radius:6px;
}
.demora-num{ font-size:22px; font-weight:700; }
.demora-pct{ font-size:15px; opacity:0.85; margin-top:2px; }
.demora-pax{ font-size:14px; opacity:0.75; margin-top:3px; }
.total-row td{
    background:rgba(255,255,255,0.15) !important;
    font-weight:900;
}
</style>

<table class="demora-table">
<tr><th>Aerolínea</th>
"""

for cat in orden_final:
    html += f"<th>{cat}</th>"
html += "<th>Total</th></tr>"

for airline,row in pivot.iloc[:-1].iterrows():
    html += f"<tr><td><b>{airline}</b></td>"
    for cat in orden_final:
        cant = int(row[f"CANTIDAD_{cat}"])
        pax = int(row[f"PAX_{cat}"])
        pct = float(row[f"PCT_{cat}"])

        color = color_for_category(cat)

        html += f"""
        <td style="background:{color}22">
            <div class="demora-num">{cant:,}</div>
            <div class="demora-pct">{pct:.1%}</div>
            <div class="demora-pax">{pax:,} pax</div>
        </td>
        """
    html += f"""
    <td>
        <div class="demora-num">{int(row['TOTAL']):,}</div>
        <div class="demora-pax">{int(row['TOTAL_PAX']):,} pax</div>
    </td></tr>
    """

# Total global
t = pivot.loc["TOTAL (TODAS)"]
html += "<tr class='total-row'><td><b>Total</b></td>"

for cat in orden_final:
    html += f"""
    <td>
        <div class="demora-num">{int(t[f"CANTIDAD_{cat}"]):,}</div>
        <div class="demora-pct">{t[f"PCT_{cat}"]:.1%}</div>
        <div class="demora-pax">{int(t[f"PAX_{cat}"]):,} pax</div>
    </td>
    """

html += f"""
<td>
    <div class="demora-num">{int(t['TOTAL']):,}</div>
    <div class="demora-pax">{int(t['TOTAL_PAX']):,} pax</div>
</td></tr></table>
"""

st.html(html)



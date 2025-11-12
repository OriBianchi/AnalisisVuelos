# streamlit_app/home.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Operaciones Aéreas - Overview", layout="wide")

# === Rutas robustas ===
# BASE_DIR: carpeta donde está este archivo (funciona desde systemd o consola)
BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR: primero busca ./data junto al archivo; si no, ../data (útil si mueves páginas)
DATA_DIR = (BASE_DIR / "data") if (BASE_DIR / "data").exists() else (BASE_DIR.parent / "data")

@st.cache_data(show_spinner=True)
def load_data():
    try:
        # Ajusta parse_dates a columnas seguras (Fecha existe en tus CSV)
        eda = pd.read_csv(DATA_DIR / "vuelos_eda.csv",
                          low_memory=False,
                          parse_dates=["Fecha"])
        aero = pd.read_csv(DATA_DIR / "aeropuertos_eda.csv", low_memory=False)

        # Limpieza mínima
        # Algunas columnas pueden venir con nombres ligeramente distintos;
        # usamos get con fallback para evitar KeyError.
        if "Aerolinea Nombre" in eda.columns:
            eda["Aerolínea"] = eda["Aerolinea Nombre"].astype(str).str.strip().str.upper()
        elif "Aerolinea" in eda.columns:
            eda["Aerolínea"] = eda["Aerolinea"].astype(str).str.strip().str.upper()
        else:
            eda["Aerolínea"] = "DESCONOCIDA"

        if "Aeropuerto" in eda.columns:
            eda["Aeropuerto"] = eda["Aeropuerto"].astype(str).str.strip().str.upper()
        else:
            eda["Aeropuerto"] = "DESCONOCIDO"

        # Mes para series temporales
        eda["mes"] = eda["Fecha"].dt.to_period("M").dt.to_timestamp()

        return eda, aero
    except FileNotFoundError as e:
        st.error(f"❌ No encontré el archivo **{e.filename}**. Busqué en **{DATA_DIR}**.")
        raise
    except Exception as e:
        st.error(f"❌ Error cargando datos desde {DATA_DIR}: {e}")
        raise

eda, aero = load_data()

# ================= Sidebar - filtros =================
st.sidebar.header("Filtros")

# Rango de fechas (con fallback por si el df está vacío)
if not eda.empty and "Fecha" in eda.columns:
    date_min, date_max = eda["Fecha"].min().date(), eda["Fecha"].max().date()
    rango = st.sidebar.date_input("Rango de fecha", (date_min, date_max))
else:
    rango = None

tipo_opts = sorted(eda["Tipo Movimiento"].dropna().unique()) if "Tipo Movimiento" in eda.columns else []
aerop_opts = sorted(eda["Aeropuerto"].dropna().unique())
aero_opts  = sorted(eda["Aerolínea"].dropna().unique())

tipo = st.sidebar.multiselect("Tipo de movimiento", options=tipo_opts)
aeropuerto = st.sidebar.multiselect("Aeropuerto", options=aerop_opts)
aerolinea = st.sidebar.multiselect("Aerolínea", options=aero_opts)

# ================= Filtrado =================
df = eda.copy()
if rango and isinstance(rango, (list, tuple)) and len(rango) == 2:
    start = pd.to_datetime(rango[0])
    end   = pd.to_datetime(rango[1])
    df = df[(df["Fecha"] >= start) & (df["Fecha"] <= end)]

if tipo:
    df = df[df["Tipo Movimiento"].isin(tipo)]
if aeropuerto:
    df = df[df["Aeropuerto"].isin(aeropuerto)]
if aerolinea:
    df = df[df["Aerolínea"].isin(aerolinea)]

# ================= KPIs =================
vuelos = int(len(df))
pax = df["Pasajeros"].sum() if "Pasajeros" in df.columns else np.nan
pax_x_vuelo = (pax / vuelos) if vuelos > 0 and pd.notna(pax) else np.nan
top_aerolinea = (df.groupby("Aerolínea").size().sort_values(ascending=False).head(1).index[0]
                 if not df.empty else "—")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Vuelos", f"{vuelos:,}")
k2.metric("Pasajeros", f"{int(pax):,}" if pd.notna(pax) else "—")
k3.metric("PAX por vuelo", f"{pax_x_vuelo:,.1f}" if pd.notna(pax_x_vuelo) else "—")
k4.metric("Top Aerolínea (por vuelos)", top_aerolinea)

# ================= Gráficos =================
# Trend mensual
if not df.empty:
    m = df.groupby("mes", as_index=False).size().rename(columns={"size": "vuelos"})
    fig_trend = px.line(m, x="mes", y="vuelos", title="Evolución mensual de vuelos")
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No hay datos para el filtro seleccionado.")

# Share por aerolínea
if not df.empty:
    share = (df.groupby("Aerolínea", as_index=False).size()
               .rename(columns={"size": "vuelos"})
               .sort_values("vuelos", ascending=False)
               .head(15))
    fig_share = px.bar(share, x="Aerolínea", y="vuelos", title="Top 15 Aerolíneas por vuelos")
    st.plotly_chart(fig_share, use_container_width=True)

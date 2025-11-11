# streamlit_app/Home.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Operaciones Aéreas - Overview", layout="wide")

@st.cache_data
def load_data():
    base = Path(__file__).parent / "data"
    eda = pd.read_csv(base / "vuelos_eda.csv", low_memory=False, parse_dates=["Fecha","Fecha_Hora_Local"])
    aero = pd.read_csv(base / "aeropuertos_eda.csv", low_memory=False)
    # Limpieza mínima
    eda["Aerolínea"] = eda["Aerolinea Nombre"].str.strip().str.upper()
    eda["Aeropuerto"] = eda["Aeropuerto"].str.strip().str.upper()
    eda["mes"] = eda["Fecha"].dt.to_period("M").dt.to_timestamp()
    return eda, aero

eda, aero = load_data()

# Sidebar - filtros
st.sidebar.header("Filtros")
date_min, date_max = eda["Fecha"].min(), eda["Fecha"].max()
rango = st.sidebar.date_input("Rango de fecha", (date_min, date_max))
tipo = st.sidebar.multiselect("Tipo de movimiento", options=sorted(eda["Tipo Movimiento"].unique()), default=None)
aeropuerto = st.sidebar.multiselect("Aeropuerto", options=sorted(eda["Aeropuerto"].unique()), default=None)
aerolinea = st.sidebar.multiselect("Aerolínea", options=sorted(eda["Aerolínea"].unique()), default=None)

df = eda.copy()
if rango:
    start, end = pd.to_datetime(rango[0]), pd.to_datetime(rango[-1])
    df = df[(df["Fecha"] >= start) & (df["Fecha"] <= end)]
if tipo: df = df[df["Tipo Movimiento"].isin(tipo)]
if aeropuerto: df = df[df["Aeropuerto"].isin(aeropuerto)]
if aerolinea: df = df[df["Aerolínea"].isin(aerolinea)]

# KPIs
vuelos = len(df)
pax = df["Pasajeros"].sum() if "Pasajeros" in df.columns else np.nan
pax_x_vuelo = pax / vuelos if vuelos > 0 and pd.notna(pax) else np.nan
top_aerolinea = (df.groupby("Aerolínea").size().sort_values(ascending=False).head(1).index[0]
                 if not df.empty else "—")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Vuelos", f"{vuelos:,}")
k2.metric("Pasajeros", f"{int(pax):,}" if pd.notna(pax) else "—")
k3.metric("PAX por vuelo", f"{pax_x_vuelo:,.1f}" if pd.notna(pax_x_vuelo) else "—")
k4.metric("Top Aerolínea (por vuelos)", top_aerolinea)

# Trend
m = df.groupby("mes").size().reset_index(name="vuelos")
fig_trend = px.line(m, x="mes", y="vuelos", title="Evolución mensual de vuelos")
st.plotly_chart(fig_trend, use_container_width=True)

# Share por aerolínea
share = (df.groupby("Aerolínea").size().reset_index(name="vuelos")
         .sort_values("vuelos", ascending=False).head(15))
fig_share = px.bar(share, x="Aerolínea", y="vuelos", title="Top 15 Aerolíneas por vuelos")
st.plotly_chart(fig_share, use_container_width=True)

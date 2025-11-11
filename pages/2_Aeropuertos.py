# streamlit_app/pages/2_Aeropuertos.py
import streamlit as st, pandas as pd, plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Aeropuertos", layout="wide")

@st.cache_data
@st.cache_data
def load_data():
    base = Path(__file__).resolve().parents[1] / "data"
    eda = pd.read_csv(base / "vuelos_eda.csv", low_memory=False, parse_dates=["Fecha","Fecha_Hora_Local"])
    aero = pd.read_csv(base / "aeropuertos_eda.csv", low_memory=False)
    eda["Aeropuerto"] = eda["Aeropuerto"].str.strip().str.upper()
    aero["denominacion"] = aero["denominacion"].str.strip().str.upper()
    return eda, aero

eda, aero = load_data()

st.sidebar.header("Filtros")
airport = st.sidebar.multiselect("Aeropuerto", sorted(eda["Aeropuerto"].unique()))

df = eda.copy()
if airport: df = df[df["Aeropuerto"].isin(airport)]

k1, k2 = st.columns(2)
k1.metric("Aeropuertos con operación", f"{df['Aeropuerto'].nunique():,}")
k2.metric("Vuelos", f"{len(df):,}")

rank = df.groupby("Aeropuerto")["Pasajeros"].sum().reset_index().sort_values("Pasajeros", ascending=False).head(20)
aero_map = (rank.merge(aero, left_on="Aeropuerto", right_on="denominacion", how="left")
                 .dropna(subset=["latitud","longitud"]))

fig_map = px.scatter_mapbox(aero_map, lat="latitud", lon="longitud", size="Pasajeros",
                            hover_name="Aeropuerto", zoom=3, height=550)
fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=40,b=0))
st.plotly_chart(fig_map, use_container_width=True)

fig_rank = px.bar(rank, x="Pasajeros", y="Aeropuerto", orientation="h", title="Top 20 Aeropuertos por PAX")
st.plotly_chart(fig_rank, use_container_width=True)

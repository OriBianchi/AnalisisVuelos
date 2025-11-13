# -*- coding: utf-8 -*-
# pages/4_Correlacion.py

import base64
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from PIL import Image
from sklearn.linear_model import LinearRegression
import catboost
from catboost import CatBoostRegressor

# ===== Tema (opcional) =====
try:
    from ui_theme import init_theme
    init_theme(show_toggle=False)
except:
    pass

st.set_page_config(page_title="Correlación — Regresión Lineal", layout="wide")

# ======================================================
# LOGO UADE
# ======================================================
def _uade_sidebar_header():
    logo_path = Path("assets/logos/uade.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
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
                margin: 0px 0px 60px 0px;
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

# ======================================================
# TITULO
# ======================================================
st.title("📈 Regresión Lineal en base a la Hipótesis")

# ======================================================
# CARGA DATASET
# ======================================================
DATA_ROOT = Path("data")

df_vuelos = pd.read_csv(DATA_ROOT / "vuelos_preliminar.csv", low_memory=False)
df_vuelos = df_vuelos.drop(df_vuelos.index[-1])

df_vuelos["Tiempo_Vuelo"] = pd.to_timedelta(df_vuelos["Tiempo_Vuelo"])
df_vuelos["minutos_vuelo"] = df_vuelos["Tiempo_Vuelo"].dt.total_seconds() / 60

# Reconstruir fecha/hora
df_vuelos["Fecha_Hora_Local_despegue"] = pd.to_datetime(df_vuelos["Fecha_Hora_Local_despegue"])

df_vuelos["Fecha"]      = df_vuelos["Fecha_Hora_Local_despegue"].dt.date
df_vuelos["Hora"]       = df_vuelos["Fecha_Hora_Local_despegue"].dt.hour
df_vuelos["Dia"]        = df_vuelos["Fecha_Hora_Local_despegue"].dt.day
df_vuelos["Mes"]        = df_vuelos["Fecha_Hora_Local_despegue"].dt.month
df_vuelos["dia_nombre"] = df_vuelos["Fecha_Hora_Local_despegue"].dt.day_name()
df_vuelos["dia_numero"] = df_vuelos["Fecha_Hora_Local_despegue"].dt.day
df_vuelos["mes_numero"] = df_vuelos["Fecha_Hora_Local_despegue"].dt.month

# Temporada alta
def temporada_alta_func(r):
    if (r["Mes"] == 12 and r["Dia"] >= 15) or (r["Mes"] == 1 and r["Dia"] <= 15):
        return 1
    if r["Mes"] == 7:
        return 1
    return 0

df_vuelos["temporada_alta"] = df_vuelos.apply(temporada_alta_func, axis=1)

# Periodo del día
def periodo_dia(h):
    if 5 <= h < 12: return "mañana"
    if 12 <= h < 20: return "tarde"
    return "noche"

df_vuelos["periodo_dia"] = df_vuelos["Hora"].apply(periodo_dia)

# ======================================================
# ARMAR DATASET DEL MODELO
# ======================================================
ordered_cols = [
    "Origen","Destino","Aerolinea","Aeronave","Pasajeros",
    "temporada_alta","periodo_dia","dia_nombre",
    "dia_numero","mes_numero"
]

df_raw = df_vuelos[ordered_cols + ["minutos_vuelo"]].copy()

# ======================================================
# CARGAR MODELO
# ======================================================
model_path = DATA_ROOT / "models" / "model_1.cbm"
model_1 = CatBoostRegressor()
model_1.load_model(str(model_path))

df_tiempo_vuelos = pd.read_csv(DATA_ROOT / "promedio_tiempos_vuelo.csv")

# ======================================================
# CORRELACION
# ======================================================

sample_size = st.slider("Tamaño de muestra", 300, 3000, 1000, step=100)
df_sample = df_raw.sample(n=sample_size, random_state=42)

# ======================================================
# PREPARAR X_sample
# ======================================================
X_sample = df_sample[ordered_cols].copy()

cat_cols = ["Origen","Destino","Aerolinea","Aeronave","periodo_dia","dia_nombre","dia_numero","mes_numero"]

for c in cat_cols:
    X_sample[c] = X_sample[c].astype(str)

cat_features_idx = [ordered_cols.index(c) for c in cat_cols]

# ======================================================
# PREDICCIÓN
# ======================================================
pool = catboost.Pool(X_sample, cat_features=cat_features_idx)
df_sample["predicted_minutos_vuelo"] = model_1.predict(pool)

# ======================================================
# PROMEDIO HISTORICO
# ======================================================
def get_avg(r):
    match = df_tiempo_vuelos[
        (df_tiempo_vuelos["Aerolinea"] == r["Aerolinea"]) &
        (df_tiempo_vuelos["Origen"] == r["Origen"]) &
        (df_tiempo_vuelos["Destino"] == r["Destino"])
    ]
    if match.empty:
        return None
    h,m,s = map(int, match.iloc[0]["Tiempo_Promedio"].split(":"))
    return (h*60 + m + s/60)

df_sample["actual_average_minutos_vuelo"] = df_sample.apply(get_avg, axis=1)

# ======================================================
# CALCULAR DELAY
# ======================================================
df_sample["delay_in_minutes"] = df_sample["predicted_minutos_vuelo"] - df_sample["actual_average_minutos_vuelo"]
df_filtered = df_sample.dropna(subset=["delay_in_minutes"])

# ======================================================
# REGRESIÓN LINEAL
# ======================================================
X = df_filtered["predicted_minutos_vuelo"].values.reshape(-1,1)
y = df_filtered["delay_in_minutes"].values

lr = LinearRegression()
lr.fit(X,y)
pred_lr = lr.predict(X)

coef_main = lr.coef_[0]

# ======================================================
# RESULTADO DEL MODELO
# ======================================================
# st.info(f"""
# ### Resultados de la regresión con el modelo predictivo

# **• Pendiente (slope):** `{coef_main:.4f}`  

# Una pendiente cercana a cero indica que la **duración predicha no explica las demoras**.
# En otras palabras, vuelos que el modelo predice como más largos **no tienden a demorar más**.
# """)

# # ======================================================
# # GRAFICO
# # ======================================================
# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=df_filtered["predicted_minutos_vuelo"],
#     y=df_filtered["delay_in_minutes"],
#     mode="markers",
#     name="Datos reales",
#     opacity=0.5
# ))

# fig.add_trace(go.Scatter(
#     x=df_filtered["predicted_minutos_vuelo"],
#     y=pred_lr,
#     mode="lines",
#     name="Regresión lineal",
#     line=dict(color="red")
# ))

# fig.update_layout(
#     title="Duración predicha vs Demora real",
#     xaxis_title="Duración predicha (min)",
#     yaxis_title="Demora (min)",
#     height=550
# )

# st.plotly_chart(fig, use_container_width=True)

# ======================================================
# HIPÓTESIS ALTERNATIVA
# ======================================================
st.subheader("📌 Hipótesis alternativa: duración histórica vs demora real")

st.markdown("""
El análisis alternativo utiliza los **datos históricos reales** sin intervención del modelo predictivo.
Se compara la duración real del vuelo contra el promedio histórico de su ruta para medir la demora real.
""")

df_sample["delay_real_in_minutes"] = (
    df_sample["minutos_vuelo"] - df_sample["actual_average_minutos_vuelo"]
)

df_filtered_alt = df_sample.dropna(subset=["actual_average_minutos_vuelo", "delay_real_in_minutes"])

# Regresión alternativa
X_alt = df_filtered_alt["actual_average_minutos_vuelo"].values.reshape(-1, 1)
y_alt = df_filtered_alt["delay_real_in_minutes"].values

lr_alt = LinearRegression()
lr_alt.fit(X_alt, y_alt)
pred_alt = lr_alt.predict(X_alt)

coef_alt = lr_alt.coef_[0]

# ======================================================
# RESULTADO HISTÓRICO
# ======================================================
st.info(f"""
### Resultados de la regresión utilizando los datos históricos

**• Pendiente (slope):** `{coef_alt:.4f}`  

Una pendiente cercana a cero indica que la **duración promedio histórica no explica las demoras reales**.
Las rutas más largas **no presentan mayores demoras respecto a su propio promedio**.
""")

# ======================================================
# GRAFICO ALTERNATIVO
# ======================================================
fig_alt = go.Figure()

fig_alt.add_trace(go.Scatter(
    x=df_filtered_alt["actual_average_minutos_vuelo"],
    y=df_filtered_alt["delay_real_in_minutes"],
    mode="markers",
    name="Datos reales",
    opacity=0.55
))

fig_alt.add_trace(go.Scatter(
    x=df_filtered_alt["actual_average_minutos_vuelo"],
    y=pred_alt,
    mode="lines",
    name="Regresión lineal",
    line=dict(color="red")
))

fig_alt.update_layout(
    title="📉 Correlación entre duración promedio histórica y demora real",
    xaxis_title="Duración promedio histórica (min)",
    yaxis_title="Demora real (min)",
    height=550
)

st.plotly_chart(fig_alt, use_container_width=True)

# ======================================================
# CONCLUSIÓN FINAL
# ======================================================
st.info("""
### 📝 Conclusión resumida

Los análisis de correlación muestran que **la duración de un vuelo no está linealmente relacionada con sus demoras**.  
Tanto la regresión basada en el **modelo predictivo**, como la realizada con la **duración histórica real**, 
arrojaron pendientes muy cercanas a cero.

Esto confirma que los vuelos más largos **no tienden a presentar mayores demoras**.  
Se puede concluir que un vuelo más extenso no implica necesariamente un mayor riesgo de retraso.
""")

# pages/3_Calcular_Demoras.py
# -*- coding: utf-8 -*-
import base64
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ===== Tema (opcional) =====
try:
    from ui_theme import init_theme
    init_theme(show_toggle=False)
except Exception:
    pass

st.set_page_config(page_title="Calcular demoras / duración estimada", layout="wide")

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
        /* Logo arriba del todo, pegado al borde superior */
        [data-testid="stSidebarNav"] {{
            background-image: url("data:image/png;base64,{img_b64}");
            background-repeat: no-repeat;
            background-position: 16px 8px;   /* más arriba */
            background-size: 120px auto;
            padding-top: 75px;              /* menos espacio debajo del logo */
        }}

        /* Bloque de texto del grupo, con línea vertical y márgenes chicos */
        [data-testid="stSidebarNav"]::before {{
            white-space: pre-line;
            display: block;

            /* Márgenes ajustados */
            margin: 0px 0px 60px 0px;
            padding-left: 10px;

            border-left: 1px solid rgba(255,255,255,0.35);
            font-size: 12px;
            color: #e5e7eb;
            content: "Grupo 1 - Ciencia de Datos\\AReinaldo Barreto\\AOriana Bianchi\\A Federico Rey\\AGabriel Kraus";
        }}
|
        </style>
        """,
        unsafe_allow_html=True,
    )
_uade_sidebar_header()

# ===== Paths y constantes =====
HERE = Path(__file__).resolve()
APP_ROOT = HERE.parents[1]
MODEL_PATH = APP_ROOT / "data" / "models" / "model_1.cbm"
DATA_PROM = APP_ROOT / "data" / "promedio_tiempos_vuelo.csv"
RUTAS_TRAIN_PATH = APP_ROOT / "data" / "models" / "rutas_train.csv"
AEROPUERTOS_PATH = APP_ROOT / "data" / "aeropuertos_detalle.csv"

POPULAR_CANON = ["AEROLINEAS ARGENTINAS SA", "FLYBONDI", "JETSMART"]

LOGOS_DIR = APP_ROOT / "assets" / "logos"
AIRLINE_LOGOS = {
    "AEROLINEAS ARGENTINAS SA": LOGOS_DIR / "aerolineas_argentinas.png",
    "FLYBONDI": LOGOS_DIR / "flybondi.png",
    "JETSMART": LOGOS_DIR / "jetsmart.png",
}

# Umbral de "entrenamiento suficiente" para una ruta
MIN_TRAIN_SAMPLES = 10  # ajustable

# ===== Helpers =====
def _normalize_airline(a: str) -> str:
    if not isinstance(a, str):
        return a
    x = a.strip().upper()
    if "AEROLINEAS" in x:
        return "AEROLINEAS ARGENTINAS SA"
    if "FLYBONDI" in x:
        return "FLYBONDI"
    if "JETSMART" in x or "JET SMART" in x:
        return "JETSMART"
    return x


def _to_minutes(td_str: str) -> float:
    td = pd.to_timedelta(td_str, errors="coerce")
    return float((td.total_seconds() / 60.0)) if pd.notna(td) else np.nan


@st.cache_resource(show_spinner=True)
def load_model():
    m = CatBoostRegressor()
    m.load_model(MODEL_PATH)
    return m


@st.cache_data(show_spinner=True)
def load_promedios_filtrados():
    df = pd.read_csv(DATA_PROM)
    # columnas esperadas: Aerolinea, Origen, Destino, Tiempo_Promedio
    expected = {"Aerolinea", "Origen", "Destino", "Tiempo_Promedio"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en promedio_tiempos_vuelo.csv: {missing}")

    df["Aerolinea_norm"] = df["Aerolinea"].apply(_normalize_airline)
    df = df[df["Aerolinea_norm"].isin(POPULAR_CANON)].copy()
    df["prom_min"] = df["Tiempo_Promedio"].apply(_to_minutes)
    df_idx = df.set_index(["Aerolinea_norm", "Origen", "Destino"]).sort_index()
    return df, df_idx


@st.cache_data(show_spinner=False)
def load_rutas_train():
    """Carga rutas que el modelo vio en entrenamiento y su n_train."""
    if not RUTAS_TRAIN_PATH.exists():
        return pd.DataFrame(), {}

    df = pd.read_csv(RUTAS_TRAIN_PATH)
    # esperamos columnas: Origen, Destino, Aerolinea, n_train (y opcionalmente ruta)
    expected = {"Origen", "Destino", "Aerolinea", "n_train"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en rutas_train.csv: {missing}")

    df["Aerolinea"] = df["Aerolinea"].apply(_normalize_airline)
    idx = df.set_index(["Aerolinea", "Origen", "Destino"])["n_train"]
    rutas_map = idx.to_dict()
    return df, rutas_map


@st.cache_data(show_spinner=True)
def load_airports():
    """
    Carga aeropuertos_detalle.csv y arma un mapa:
      código (local) -> "código — Denominación"
    Usamos la columna 'local' porque es la que se usa en Origen/Destino
    (AER, BAR, etc.). Si no existiera, cae a 'iata'.
    """
    if not AEROPUERTOS_PATH.exists():
        return pd.DataFrame(), {}

    # El archivo viene separado por ';'
    df = pd.read_csv(AEROPUERTOS_PATH, sep=";")

    # Normalizamos nombres de columnas a minúsculas
    df.columns = [c.lower() for c in df.columns]

    # Preferimos 'local' (códigos AER, BAR, etc.), si no existe usamos 'iata'
    key_col = "local" if "local" in df.columns else "iata"

    if key_col not in df.columns or "denominacion" not in df.columns:
        return df, {}

    df[key_col] = df[key_col].astype(str).str.strip()
    df["denominacion"] = df["denominacion"].astype(str).str.strip()

    # Quitamos filas sin código
    df = df[df[key_col] != ""]

    # Texto a mostrar: "AER — AERÓDROMO TAL"
    df["display"] = df[key_col] + " — " + df["denominacion"]

    mapping = dict(zip(df[key_col], df["display"]))
    return df, mapping


def airport_label(iata: str) -> str:
    """Devuelve 'IATA — Nombre completo' si existe, sino el código."""
    if not isinstance(iata, str):
        return iata
    return airport_name_map.get(iata, iata)


# ===== Carga artefactos =====
model = load_model()
df_prom, df_prom_idx = load_promedios_filtrados()
df_rutas_train, rutas_train_map = load_rutas_train()
df_airports, airport_name_map = load_airports()

# ===== Lógica de negocio =====
def calcular_demora_modelo(
    aerolinea,
    origen,
    destino,
    periodo_dia="mañana",
    dia_nombre="Monday",
    dia_numero=15,
    mes_numero=7,
):
    """
    Reglas:
      - Si la ruta no aparece en rutas_train.csv o tiene pocos registros de entrenamiento,
        usamos SOLO el promedio histórico (prom_min) -> evita predicciones locas.
      - Si tiene entrenamiento suficiente, usamos la predicción del modelo.
    """
    # 1) promedio histórico desde promedio_tiempos_vuelo.csv
    try:
        prom_min = float(
            np.round(df_prom_idx.loc[(aerolinea, origen, destino)]["prom_min"], 0)
        )
    except KeyError:
        # esa aerolínea no opera la ruta según el CSV de promedios
        raise ValueError("Ruta no está en la tabla de promedios para esa aerolínea.")

    # 2) cuántas veces se entrenó esa ruta
    n_train = rutas_train_map.get((aerolinea, origen, destino), 0)

    # 3) si no tuvo entrenamiento suficiente, devolvemos el promedio
    if n_train < MIN_TRAIN_SAMPLES:
        pred_min = prom_min
        diff = 0.0
        return pred_min, prom_min, diff

    # 4) si tuvo entrenamiento suficiente, usamos el modelo
    entrada = {
        "Origen": origen,
        "Destino": destino,
        "Aerolinea": aerolinea,
        "periodo_dia": periodo_dia,
        "dia_nombre": dia_nombre,
        "dia_numero": int(dia_numero),
        "mes_numero": int(mes_numero),
    }
    pred_min = float(np.round(model.predict(pd.DataFrame([entrada]))[0], 0))
    diff = float(np.round(pred_min - prom_min, 0))
    return pred_min, prom_min, diff


# ======================================================
#                 INTERFAZ PRINCIPAL
# ======================================================
st.title("🛫 Calcular demoras — por ruta o vuelo puntual")

modo = st.sidebar.radio(
    "Seleccioná modo de análisis:",
    [
        "Por ruta (todas las aerolíneas)",
        "Detalle puntual (una aerolínea)",
    ],
    index=0,
)

# ------------------------------------------------------
# 🔹 1) Comparativa por ruta (todas las aerolíneas)
# ------------------------------------------------------
if modo == "Por ruta (todas las aerolíneas)":
    st.subheader("Comparativa por ruta — promedio vs modelo")

    # Combos de ruta basados en todas las rutas existentes (Top 3 aerolíneas)
    orig_opts = sorted(df_prom["Origen"].dropna().unique().tolist())
    origen = st.selectbox("Origen", orig_opts, format_func=airport_label)

    dest_opts = sorted(
        df_prom.loc[df_prom["Origen"] == origen, "Destino"].dropna().unique().tolist()
    )
    destino = st.selectbox("Destino", dest_opts, format_func=airport_label)

    if st.button(
        "Comparar duración promedio vs modelo por aerolínea",
        type="primary",
        use_container_width=True,
    ):

        resultados = []
        for aer in POPULAR_CANON:
            if (aer, origen, destino) not in df_prom_idx.index:
                continue

            try:
                pred, prom, diff = calcular_demora_modelo(aer, origen, destino)
            except Exception:
                continue

            resultados.append(
                {
                    "Aerolinea": aer,
                    "Promedio": prom,
                    "Predicho (modelo)": pred,
                    "Diferencia (min)": diff,
                }
            )

        if not resultados:
            st.warning(
                "Ninguna de las 3 aerolíneas opera esa ruta o no hay datos suficientes."
            )
            st.stop()

        df_cmp = pd.DataFrame(resultados)

        # ============================
        # Situación / Semáforo
        # ============================
        TOL = 10

        def clasificar(diff):
            if diff > TOL:
                return "Probable demora"
            elif diff < -TOL:
                return "Más rápido que el promedio"
            else:
                return "En promedio"

        df_cmp["Situación"] = df_cmp["Diferencia (min)"].apply(clasificar)

        # ============================
        # Tabla final — (Promedio → Predicho)
        # ============================
        df_cmp = df_cmp[
            ["Aerolinea", "Promedio", "Predicho (modelo)", "Diferencia (min)", "Situación"]
        ]

        # Colores suaves
        def color_situacion(val):
            if val == "Probable demora":
                return "color: #b91c1c; font-weight: bold;"  # rojo suave
            elif val == "En promedio":
                return "color: #1e3a8a; font-weight: bold;"  # azul suave
            elif val == "Más rápido que el promedio":
                return "color: #047857; font-weight: bold;"  # verde suave
            return ""

        styler = (
            df_cmp.style.format(
                {
                    "Promedio": "{:.1f}",
                    "Predicho (modelo)": "{:.1f}",
                    "Diferencia (min)": "{:.1f}",
                }
            ).applymap(color_situacion, subset=["Situación"])
        )

        st.dataframe(styler, use_container_width=True)

        # ============================
        # Gráfico de barras con logos
        # ============================
        # Colores de predicción por situación
        palette = {
            "Probable demora": "#fca5a5",
            "En promedio": "#93c5fd",
            "Más rápido que el promedio": "#6ee7b7",
        }
        pred_colors = df_cmp["Situación"].map(palette)

        promedio_color = "#d4d4d8"  # gris suave

        fig = go.Figure()

        # Barras de promedio
        fig.add_bar(
            name="Promedio",
            x=df_cmp["Aerolinea"],
            y=df_cmp["Promedio"],
            marker_color=promedio_color,
        )

        # Barras de predicho
        fig.add_bar(
            name="Predicho (modelo)",
            x=df_cmp["Aerolinea"],
            y=df_cmp["Predicho (modelo)"],
            marker_color=pred_colors,
        )

        max_y = float(
            max(df_cmp["Promedio"].max(), df_cmp["Predicho (modelo)"].max())
        )
        fig.update_yaxes(range=[0, max_y * 1.30])

        fig.update_layout(
            barmode="group",
            title=(
                "Duración promedio vs modelo — Ruta "
                f"{airport_label(origen)} → {airport_label(destino)}"
            ),
            xaxis_title="Aerolínea",
            yaxis_title="Minutos",
            legend_title="Serie",
        )

        # ============================
        # Logos sobre las barras predichas
        # ============================
        for _, row in df_cmp.iterrows():
            aer = row["Aerolinea"]
            pred_val = row["Predicho (modelo)"]

            logo_path = AIRLINE_LOGOS.get(aer)
            if not logo_path or not logo_path.exists():
                continue

            img = Image.open(str(logo_path))

            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x",
                    yref="y",
                    x=aer,
                    y=pred_val + (max_y * 0.05),  # ⭐ Encima de la barra
                    sizex=0.5,
                    sizey=max_y * 0.15,
                    xanchor="center",
                    yanchor="bottom",
                    layer="above",
                )
            )

        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# 🔹 2) Detalle puntual (una aerolínea)
# ------------------------------------------------------
elif modo == "Detalle puntual (una aerolínea)":
    st.subheader("Detalle puntual — seleccionar una aerolínea y ruta")

    aerolinea_sel = st.selectbox("Aerolínea", POPULAR_CANON)

    orig_opts = (
        df_prom.loc[df_prom["Aerolinea_norm"] == aerolinea_sel, "Origen"]
        .dropna()
        .unique()
        .tolist()
    )
    orig_opts = sorted(orig_opts)
    origen = st.selectbox("Origen", orig_opts, format_func=airport_label)

    dest_opts = (
        df_prom.loc[
            (df_prom["Aerolinea_norm"] == aerolinea_sel)
            & (df_prom["Origen"] == origen),
            "Destino",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    dest_opts = sorted(dest_opts)
    destino = st.selectbox("Destino", dest_opts, format_func=airport_label)

    periodo = st.selectbox("Periodo del día", ["mañana", "tarde", "noche"], index=0)
    dia_numero = st.number_input("Día del mes", 1, 31, 15, 1)
    mes_numero = st.number_input("Mes (número)", 1, 12, 7, 1)

    if st.button(
        "Calcular duración estimada", type="primary", use_container_width=True
    ):
        try:
            pred, prom, diff = calcular_demora_modelo(
                aerolinea_sel, origen, destino, periodo, "Monday", dia_numero, mes_numero
            )
        except Exception as e:
            st.error(f"No se pudo calcular: {e}")
        else:
            st.markdown(
                f"### Ruta seleccionada: {airport_label(origen)} → {airport_label(destino)}"
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Duración estimada (modelo/negocio)", f"{int(pred)} min")
            c2.metric("Promedio histórico", f"{int(prom)} min")
            c3.metric("Diferencia", f"{int(diff)} min")

            if diff > 10:
                st.error(
                    f"⚠️ Se predice una demora en el vuelo de **{int(diff)} minutos**."
                )
            elif diff < -10:
                st.success(
                    f"Se predice que el vuelo será más rápido que el promedio por **{int(abs(diff))} minutos**."
                )
            else:
                st.info(
                    "Se predice que el vuelo durará dentro de ±10 minutos del promedio histórico."
                )

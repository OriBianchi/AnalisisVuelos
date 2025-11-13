import pandas as pd

df = pd.read_csv("data/vuelos_preliminar.csv", low_memory=False)

total_filas = len(df)
unicos_por_id = df["Vuelo_ID"].nunique(dropna=True)
nulos_id = df["Vuelo_ID"].isna().sum()
duplicados_id = df.duplicated(subset=["Vuelo_ID"]).sum()  # filas extra con el mismo ID

print("Filas totales:", total_filas)
print("Vuelos únicos (Vuelo_ID):", unicos_por_id)
print("Filas con Vuelo_ID nulo:", nulos_id)
print("Filas con Vuelo_ID duplicado:", duplicados_id)
print("Comprobación: tot ≈ únicos + nulos + duplicados ?",
      total_filas, "≈", unicos_por_id + nulos_id + duplicados_id)

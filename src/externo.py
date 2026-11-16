"""Carga del conjunto externo: reseñas reales de Google Maps de negocios colombianos.

Formato esperado (data/externo/resenas_google_colombia.csv, UTF-8):
    texto, estrellas, etiqueta, negocio, ciudad, tipo_negocio
- `etiqueta` es la etiqueta MANUAL (negativo / neutral / positivo), asignada leyendo
  el texto, no copiada de las estrellas.
- `estrellas` (1-5) permite comparar la etiqueta manual con la regla de estrellas.
"""
import pandas as pd

from config import DATA_EXTERNO, LABEL2ID


def cargar_externo() -> pd.DataFrame | None:
    if not DATA_EXTERNO.exists():
        return None
    df = pd.read_csv(DATA_EXTERNO)
    df = df.dropna(subset=["texto", "etiqueta"])
    df["etiqueta"] = df["etiqueta"].astype(str).str.strip().str.lower()
    df = df[df["etiqueta"].isin(LABEL2ID)]
    if df.empty:
        return None
    df["label"] = df["etiqueta"].map(LABEL2ID)
    df["texto"] = df["texto"].astype(str).map(lambda t: " ".join(t.split()))
    return df.reset_index(drop=True)

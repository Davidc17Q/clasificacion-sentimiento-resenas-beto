"""Configuración compartida del proyecto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw" / "Rest-Mex_2025_train.csv"
DATA_PROC = RAIZ / "data" / "processed"
DATA_EXTERNO = RAIZ / "data" / "externo" / "resenas_google_colombia.csv"
RESULTADOS = RAIZ / "resultados"
MODELOS = RAIZ / "modelos"

SEMILLA = 42

# Agrupación de estrellas en tres clases de sentimiento.
#   1-2 estrellas -> negativo, 3 -> neutral, 4-5 -> positivo
ETIQUETAS = ["negativo", "neutral", "positivo"]
ID2LABEL = {i: e for i, e in enumerate(ETIQUETAS)}
LABEL2ID = {e: i for i, e in enumerate(ETIQUETAS)}


def estrellas_a_clase(estrellas: float) -> int:
    if estrellas <= 2:
        return 0
    if estrellas == 3:
        return 1
    return 2


# Corpus balanceado: N reseñas por clase (la clase negativa limita: ~10.9k).
POR_CLASE = 10_000

MODELOS_HF = {
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "mdistilbert": "distilbert-base-multilingual-cased",
}

"""Etapa 1 — Construcción y preprocesamiento del corpus.

- Carga Rest-Mex 2025 (TripAdvisor, México; CC-BY-4.0).
- Repara codificación rota (mojibake tipo "CancÃºn") con ftfy.
- Une título + reseña, elimina duplicados y textos vacíos.
- Agrupa estrellas en 3 clases (1-2 neg, 3 neu, 4-5 pos).
- Submuestrea de forma balanceada (POR_CLASE por clase) y divide
  estratificado 70/15/15 (estratificando por clase y tipo de negocio).
- Guarda además un "test natural" con la distribución real (muy sesgada
  a positivos) usando reseñas que NO entraron al corpus balanceado.
"""
import json

import ftfy
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PROC, DATA_RAW, ETIQUETAS, POR_CLASE, SEMILLA, estrellas_a_clase


def limpiar(texto: str) -> str:
    texto = ftfy.fix_text(str(texto))
    return " ".join(texto.split())  # normaliza espacios / saltos de línea


def main() -> None:
    df = pd.read_csv(DATA_RAW)
    n_original = len(df)

    df["Title"] = df["Title"].fillna("")
    df["texto"] = (df["Title"].map(limpiar) + ". " + df["Review"].map(limpiar)).str.strip(". ")
    df = df[df["texto"].str.len() > 10]
    df = df.drop_duplicates(subset="texto")
    df["label"] = df["Polarity"].map(estrellas_a_clase)
    df["estrellas"] = df["Polarity"].astype(int)
    df = df.rename(columns={"Type": "tipo"})[["texto", "label", "estrellas", "tipo", "Town"]]

    dist_real = df["label"].value_counts(normalize=True).sort_index()

    # Corpus balanceado
    partes = [g.sample(n=min(POR_CLASE, len(g)), random_state=SEMILLA) for _, g in df.groupby("label")]
    bal = pd.concat(partes).sample(frac=1, random_state=SEMILLA)
    estrato = bal["label"].astype(str) + "_" + bal["tipo"]

    train, temp = train_test_split(bal, test_size=0.30, stratify=estrato, random_state=SEMILLA)
    estrato_temp = temp["label"].astype(str) + "_" + temp["tipo"]
    val, test = train_test_split(temp, test_size=0.50, stratify=estrato_temp, random_state=SEMILLA)

    # Test con distribución natural, sin reseñas usadas en el corpus balanceado
    resto = df.drop(index=bal.index)
    n_nat = 10_000
    nat_partes = []
    for lab, prop in dist_real.items():
        g = resto[resto["label"] == lab]
        nat_partes.append(g.sample(n=min(len(g), round(n_nat * prop)), random_state=SEMILLA))
    test_natural = pd.concat(nat_partes).sample(frac=1, random_state=SEMILLA)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    for nombre, parte in [("train", train), ("val", val), ("test", test), ("test_natural", test_natural)]:
        parte.to_csv(DATA_PROC / f"{nombre}.csv", index=False)

    resumen = {
        "n_original": n_original,
        "n_tras_limpieza": len(df),
        "distribucion_real": {ETIQUETAS[k]: round(v, 4) for k, v in dist_real.items()},
        "distribucion_estrellas": df["estrellas"].value_counts().sort_index().to_dict(),
        "por_clase": POR_CLASE,
        "tamanos": {"train": len(train), "val": len(val), "test": len(test), "test_natural": len(test_natural)},
        "test_natural_dist": test_natural["label"].map(lambda i: ETIQUETAS[i]).value_counts().to_dict(),
        "longitud_caracteres": bal["texto"].str.len().describe().round(1).to_dict(),
        "tipos": bal["tipo"].value_counts().to_dict(),
    }
    (DATA_PROC / "resumen_datos.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

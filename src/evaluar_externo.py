"""Etapa 3c — Evaluación de los tres modelos en el conjunto externo colombiano.

Actualiza el bloque "externo_colombia" de resultados/<corrida>/metricas.json y
resultados/tfidf.json, y guarda las predicciones de BETO por reseña con el análisis
por tipo de negocio y por ciudad.

Uso: python src/evaluar_externo.py
"""
import json

import pandas as pd
import torch
from transformers import pipeline

import metricas
import numpy as np
from sklearn.metrics import f1_score

from config import ETIQUETAS, RESULTADOS, SEMILLA
from externo import cargar_externo


def predecir(ruta, textos, max_len=128):
    clf = pipeline("text-classification", model=str(ruta), device=0 if torch.cuda.is_available() else -1,
                   truncation=True, max_length=max_len, top_k=None)
    salidas = clf(textos, batch_size=32)
    preds, probs = [], []
    for s in salidas:
        d = {x["label"]: x["score"] for x in s}
        p = [d[e] for e in ETIQUETAS]
        probs.append(p)
        preds.append(int(max(range(3), key=lambda i: p[i])))
    return preds, probs


def actualizar(json_path, bloque):
    m = json.loads(json_path.read_text(encoding="utf-8"))
    m["externo_colombia"] = bloque
    json_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ext = cargar_externo()
    if ext is None:
        raise SystemExit("No hay data/externo/resenas_google_colombia.csv con filas etiquetadas.")
    final = json.loads((RESULTADOS / "beto_final.json").read_text(encoding="utf-8"))
    textos = ext["texto"].tolist()
    resumen = {}
    df = ext.copy()
    todas = {}

    for clave, corrida in [("beto", final["corrida"]), ("mdistilbert", final["corrida_mdistilbert"])]:
        m = json.loads((RESULTADOS / corrida / "metricas.json").read_text(encoding="utf-8"))
        preds, probs = predecir(m["mejor_checkpoint"], textos, m["hiperparametros"]["max_len"])
        todas[clave] = np.array(preds)
        bloque = metricas.calcular(ext["label"], preds)
        if clave == "beto":
            df["pred"] = [ETIQUETAS[p] for p in preds]
            df[["p_neg", "p_neu", "p_pos"]] = probs
            for col in ["tipo_negocio", "ciudad"]:
                if col in df:
                    bloque[f"por_{col}"] = {
                        str(k): {"n": len(g), "accuracy": round(float((g["pred"] == g["etiqueta"]).mean()), 4)}
                        for k, g in df.groupby(col)
                    }
            # acuerdo entre la etiqueta manual y la regla de estrellas
            if "estrellas" in df:
                regla = df["estrellas"].map(lambda s: "negativo" if s <= 2 else "neutral" if s == 3 else "positivo")
                bloque["acuerdo_manual_vs_estrellas"] = round(float((regla == df["etiqueta"]).mean()), 4)
        actualizar(RESULTADOS / corrida / "metricas.json", bloque)
        resumen[clave] = bloque["f1_macro"]

    # baseline TF-IDF: se reentrena rápido para no depender de un pickle
    import baseline_tfidf
    tfidf = baseline_tfidf.main()
    todas["tfidf"] = np.asarray(tfidf.predict(ext["texto"]))
    resumen["tfidf"] = json.loads((RESULTADOS / "tfidf.json").read_text(encoding="utf-8"))["externo_colombia"]["f1_macro"]

    df["pred_mdistilbert"] = [ETIQUETAS[p] for p in todas["mdistilbert"]]
    df["pred_tfidf"] = [ETIQUETAS[p] for p in todas["tfidf"]]
    df.to_csv(RESULTADOS / "predicciones_externo.csv", index=False)

    # Bootstrap (2000 remuestreos): intervalo del 95% del F1 macro y de la diferencia BETO - otro.
    # Con ~170 reseñas, diferencias pequeñas entre modelos pueden ser solo azar del muestreo.
    y = ext["label"].to_numpy()
    rng = np.random.default_rng(SEMILLA)
    f1 = lambda yt, yp: f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0)
    muestras = {k: [] for k in todas}
    difs = {k: [] for k in todas if k != "beto"}
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        v = {k: f1(y[idx], p[idx]) for k, p in todas.items()}
        for k in todas:
            muestras[k].append(v[k])
        for k in difs:
            difs[k].append(v["beto"] - v[k])
    ic = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    resumen_ext = {
        "n": int(len(y)),
        "f1_macro": resumen,
        "ic95_f1": {k: ic(v) for k, v in muestras.items()},
        "ic95_diferencia_beto_menos": {k: ic(v) for k, v in difs.items()},
        "distribucion_etiquetas": df["etiqueta"].value_counts().to_dict(),
        "n_negocios": int(df["negocio"].nunique()), "n_ciudades": int(df["ciudad"].nunique()),
    }
    (RESULTADOS / "externo_resumen.json").write_text(json.dumps(resumen_ext, indent=2, ensure_ascii=False), encoding="utf-8")
    print("F1 macro en Colombia:", resumen)
    print("IC95:", resumen_ext["ic95_f1"], "diferencias:", resumen_ext["ic95_diferencia_beto_menos"])


if __name__ == "__main__":
    main()

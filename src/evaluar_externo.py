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
from config import ETIQUETAS, RESULTADOS
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

    for clave, corrida in [("beto", final["corrida"]), ("mdistilbert", final["corrida_mdistilbert"])]:
        m = json.loads((RESULTADOS / corrida / "metricas.json").read_text(encoding="utf-8"))
        preds, probs = predecir(m["mejor_checkpoint"], textos, m["hiperparametros"]["max_len"])
        bloque = metricas.calcular(ext["label"], preds)
        if clave == "beto":
            df = ext.copy()
            df["pred"] = [ETIQUETAS[p] for p in preds]
            df[["p_neg", "p_neu", "p_pos"]] = probs
            df.to_csv(RESULTADOS / "predicciones_externo.csv", index=False)
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
    baseline_tfidf.main()
    resumen["tfidf"] = json.loads((RESULTADOS / "tfidf.json").read_text(encoding="utf-8"))["externo_colombia"]["f1_macro"]
    print("F1 macro en Colombia:", resumen)


if __name__ == "__main__":
    main()

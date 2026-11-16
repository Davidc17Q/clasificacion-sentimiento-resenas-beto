"""Etapa 2a — Baseline clásico: TF-IDF (palabras 1-2 gramas) + Regresión logística."""
import json
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import metricas
from config import DATA_PROC, RESULTADOS
from externo import cargar_externo


def main():
    train = pd.read_csv(DATA_PROC / "train.csv")
    modelo = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True, lowercase=True),
        LogisticRegression(C=4.0, max_iter=2000),
    )
    t0 = time.time()
    modelo.fit(train["texto"], train["label"])
    salida = {"modelo": "TF-IDF + Regresión logística", "segundos_entrenamiento": round(time.time() - t0, 1)}

    for nombre in ["val", "test", "test_natural"]:
        df = pd.read_csv(DATA_PROC / f"{nombre}.csv")
        salida[nombre] = metricas.calcular(df["label"], modelo.predict(df["texto"]))
    ext = cargar_externo()
    if ext is not None:
        salida["externo_colombia"] = metricas.calcular(ext["label"], modelo.predict(ext["texto"]))

    # Palabras más influyentes por clase (interpretabilidad del baseline)
    vec, clf = modelo.named_steps["tfidfvectorizer"], modelo.named_steps["logisticregression"]
    vocab = vec.get_feature_names_out()
    salida["top_terminos"] = {
        clase: [vocab[j] for j in clf.coef_[i].argsort()[::-1][:15]]
        for i, clase in enumerate(["negativo", "neutral", "positivo"])
    }

    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "tfidf.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: (v["f1_macro"] if isinstance(v, dict) and "f1_macro" in v else v)
                      for k, v in salida.items() if k != "top_terminos"}, indent=2, ensure_ascii=False))
    return modelo


if __name__ == "__main__":
    main()

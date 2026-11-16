"""Etapa 3b — Análisis de las ETAPAS INTERMEDIAS de BETO.

Para cada una de las 13 salidas internas (embeddings + 12 capas encoder) se extrae el
vector [CLS] y se mide qué tan separables son las tres clases con una sonda lineal
(regresión logística entrenada sobre ese vector congelado). Se compara:
  - BETO preentrenado (sin fine-tuning)  -> qué "sabe" el modelo de fábrica
  - BETO ajustado (fine-tuned)           -> qué aprendió con nuestras reseñas
Además guarda proyecciones PCA 2D por capa y la atención del [CLS] en la última capa
para algunos ejemplos.

Uso: python src/analisis_capas.py --modelo_ft modelos/beto_final
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from config import DATA_PROC, MODELOS_HF, RESULTADOS, SEMILLA

N_POR_CLASE = 700
CAPAS_PCA = [1, 3, 6, 9, 12]  # la capa 0 no: ahí el [CLS] es idéntico para todas las reseñas
EJEMPLOS_ATENCION = [
    "La comida estaba deliciosa y el servicio fue excelente, sin duda volveremos.",
    "El lugar es bonito pero la atención fue lenta y la comida llegó fría.",
    "Pésima experiencia, nos cobraron de más y el mesero fue grosero.",
    "Está bien, nada del otro mundo. Precios normales.",
]


@torch.no_grad()
def cls_por_capa(ruta, textos, tok, dispositivo, max_len=256, lote=32):
    modelo = AutoModel.from_pretrained(ruta, output_hidden_states=True, attn_implementation="eager").to(dispositivo).eval()
    salidas = []
    for i in range(0, len(textos), lote):
        enc = tok(textos[i:i + lote], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dispositivo)
        hs = modelo(**enc).hidden_states  # tupla de 13 tensores [B, T, 768]
        salidas.append(torch.stack([h[:, 0, :] for h in hs], dim=1).float().cpu())  # [B, 13, 768]
    return modelo, torch.cat(salidas).numpy()


@torch.no_grad()
def atencion_cls(modelo, tok, dispositivo):
    resultado = []
    for texto in EJEMPLOS_ATENCION:
        enc = tok(texto, return_tensors="pt").to(dispositivo)
        out = modelo(**enc, output_attentions=True)
        att = out.attentions[-1][0].mean(0)[0]  # última capa, promedio de 12 cabezas, fila del [CLS]
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        resultado.append({"texto": texto, "tokens": tokens, "pesos": [round(float(x), 4) for x in att.cpu()]})
    return resultado


def sondas(X, y):
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEMILLA)
    f1s = []
    for capa in range(X.shape[1]):
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(X[idx_tr, capa], y[idx_tr])
        f1s.append(round(float(f1_score(y[idx_te], clf.predict(X[idx_te, capa]), average="macro")), 4))
    return f1s


def pca_por_capa(X, y, n_puntos=450):
    idx = np.random.RandomState(SEMILLA).choice(len(y), n_puntos, replace=False)
    res = {}
    for capa in CAPAS_PCA:
        P = PCA(n_components=2, random_state=SEMILLA).fit_transform(X[idx, capa])
        P = P / np.abs(P).max(0)
        res[str(capa)] = [[round(float(a), 3), round(float(b), 3), int(c)] for (a, b), c in zip(P, y[idx])]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo_ft", required=True)
    a = p.parse_args()
    disp = "cuda" if torch.cuda.is_available() else "cpu"

    test = pd.read_csv(DATA_PROC / "test.csv")
    muestra = pd.concat([g.sample(N_POR_CLASE, random_state=SEMILLA) for _, g in test.groupby("label")])
    textos, y = muestra["texto"].tolist(), muestra["label"].to_numpy()

    tok = AutoTokenizer.from_pretrained(a.modelo_ft)
    _, X_pre = cls_por_capa(MODELOS_HF["beto"], textos, tok, disp)
    modelo_ft, X_ft = cls_por_capa(a.modelo_ft, textos, tok, disp)

    salida = {
        "capas": ["embeddings"] + [f"capa {i}" for i in range(1, 13)],
        "sonda_f1_preentrenado": sondas(X_pre, y),
        "sonda_f1_finetuned": sondas(X_ft, y),
        "pca_finetuned": pca_por_capa(X_ft, y),
        "pca_preentrenado": pca_por_capa(X_pre, y),
        "atencion": atencion_cls(modelo_ft, tok, disp),
    }
    (RESULTADOS / "capas.json").write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    print("preentrenado:", salida["sonda_f1_preentrenado"])
    print("fine-tuned:  ", salida["sonda_f1_finetuned"])


if __name__ == "__main__":
    main()

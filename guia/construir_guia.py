"""Inyecta los resultados reales (resultados/*.json) en la plantilla de la guía.

Uso: python guia/construir_guia.py  ->  guia/guia-defensa.html
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from config import ETIQUETAS  # noqa: E402

RES = RAIZ / "resultados"


def leer(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    final = leer(RES / "beto_final.json")  # {"corrida": "...", ...}
    corrida_beto = final["corrida"]
    beto = leer(RES / corrida_beto / "metricas.json")
    mdist = leer(RES / final["corrida_mdistilbert"] / "metricas.json")
    tfidf = leer(RES / "tfidf.json")
    historial = [h for h in leer(RES / corrida_beto / "historial.json") if "epoch" in h]

    busqueda = []
    for d in sorted(RES.iterdir()):
        if d.is_dir() and d.name.startswith("beto") and (d / "metricas.json").exists():
            m = leer(d / "metricas.json")
            busqueda.append({"nombre": d.name, "lr": m["hiperparametros"]["lr"], "max_len": m["hiperparametros"]["max_len"],
                             "f1_val": m["val"]["f1_macro"], "seg": m["segundos_entrenamiento"],
                             "elegida": d.name == corrida_beto})

    # Errores representativos: los más confiados entre los mal clasificados
    import pandas as pd
    pred = pd.read_csv(RES / corrida_beto / "predicciones_test.csv")
    pred["conf"] = pred[["p_neg", "p_neu", "p_pos"]].max(axis=1)
    err = pred[pred["label"] != pred["pred"]]
    err = err[err["texto"].str.len() < 260]
    sel = pd.concat([err[(err.label == a) & (err.pred == b)].nlargest(2, "conf")
                     for a, b in [(0, 2), (2, 0), (1, 2), (1, 0), (0, 1), (2, 1)]])
    errores = [{"texto": r.texto, "real": ETIQUETAS[r.label], "pred": ETIQUETAS[r.pred], "conf": round(r.conf, 3)}
               for r in sel.itertuples()]

    datos = {
        "datos": leer(RAIZ / "data" / "processed" / "resumen_datos.json"),
        "modelos": {"beto": beto, "mdistilbert": mdist, "tfidf": tfidf},
        "historial_beto": historial,
        "busqueda": busqueda,
        "errores": errores,
        "capas": leer(RES / "capas.json") if (RES / "capas.json").exists() else None,
    }
    if datos["capas"]:
        datos["capas"]["n_muestras"] = 2100
    plantilla = (RAIZ / "guia" / "plantilla.html").read_text(encoding="utf-8")
    js = json.dumps(datos, ensure_ascii=False).replace("</", "<\\/")
    salida = RAIZ / "guia" / "guia-defensa.html"
    salida.write_text(plantilla.replace("/*__DATOS__*/", js), encoding="utf-8")
    print("Guía generada:", salida, f"({salida.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()

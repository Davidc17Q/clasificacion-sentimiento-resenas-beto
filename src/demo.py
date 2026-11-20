"""Demo en vivo: escribe una reseña y el BETO ajustado dice si es negativa, neutral o positiva.

Uso:
    python src/demo.py              -> modo interactivo (escribe reseñas; Enter vacío o "salir" para terminar)
    python src/demo.py --ejemplos   -> corre primero las frases preparadas para la sustentación
    python src/demo.py "texto"      -> clasifica solo ese texto

Usa el mejor checkpoint de BETO (resultados/beto_final.json -> metricas.json -> mejor_checkpoint).
No necesita internet: el modelo está guardado en modelos/_checkpoints/.
"""
import json
import logging
import os
import sys
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.disable(logging.WARNING)  # oculta avisos técnicos de torch/transformers durante la demo
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.disable_progress_bar()

from config import ETIQUETAS, RESULTADOS  # noqa: E402

# Frases pensadas para mostrar aciertos claros, casos difíciles y una contradicción con las estrellas
EJEMPLOS = [
    "La comida llegó fría, el mesero fue grosero y nos cobraron de más. No vuelvo.",
    "Excelente atención, el corte quedó perfecto y el lugar es muy limpio. ¡Recomendadísimo!",
    "Normal, nada del otro mundo. La comida estaba bien pero un poco cara para lo que es.",
    "El café es bueno, aunque la atención fue algo lenta.",
    "Qué maravilla esperar una hora para que te traigan el plato equivocado.",  # sarcasmo
    "No estuvo mal.",
]

COLORES = {"negativo": "\033[91m", "neutral": "\033[93m", "positivo": "\033[92m"}
FIN = "\033[0m"


def cargar():
    final = json.loads((RESULTADOS / "beto_final.json").read_text(encoding="utf-8"))
    m = json.loads((RESULTADOS / final["corrida"] / "metricas.json").read_text(encoding="utf-8"))
    ruta = m["mejor_checkpoint"]
    if not os.path.isdir(ruta):
        raise SystemExit(f"No encuentro el modelo en {ruta}. Hay que entrenarlo primero (src/entrenar_transformer.py).")
    print(f"Cargando BETO ajustado ({final['corrida']}) ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta).eval()
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(disp)
    print(f"Listo en {time.time() - t0:.1f} s (usando {disp.upper()}).\n")
    return tok, modelo, disp, m["hiperparametros"]["max_len"]


@torch.no_grad()
def clasificar(texto, tok, modelo, disp, max_len):
    x = tok(texto, truncation=True, max_length=max_len, return_tensors="pt").to(disp)
    probs = torch.softmax(modelo(**x).logits, dim=-1)[0].tolist()
    return probs, len(x["input_ids"][0])


def mostrar(texto, tok, modelo, disp, max_len):
    t0 = time.time()
    probs, n_tokens = clasificar(texto, tok, modelo, disp, max_len)
    ms = (time.time() - t0) * 1000
    gana = ETIQUETAS[max(range(3), key=lambda i: probs[i])]
    print(f"  Reseña: {texto}")
    print(f"  -> {COLORES[gana]}{gana.upper()}{FIN}   ({n_tokens} tokens, {ms:.0f} ms)")
    for e, p in zip(ETIQUETAS, probs):
        barra = "█" * round(p * 30)
        print(f"     {e:<9} {COLORES[e]}{barra:<30}{FIN} {p * 100:5.1f}%")
    print()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    os.system("")  # activa los colores ANSI en la consola de Windows
    args = sys.argv[1:]
    tok, modelo, disp, max_len = cargar()

    textos = [a for a in args if a != "--ejemplos"]
    if textos:
        for t in textos:
            mostrar(t, tok, modelo, disp, max_len)
        return
    if "--ejemplos" in args:
        print("=== Frases preparadas ===\n")
        for t in EJEMPLOS:
            mostrar(t, tok, modelo, disp, max_len)

    print("Escribe una reseña y pulsa Enter (Enter vacío o 'salir' para terminar).\n")
    while True:
        try:
            t = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not t or t.lower() == "salir":
            break
        mostrar(t, tok, modelo, disp, max_len)


if __name__ == "__main__":
    main()

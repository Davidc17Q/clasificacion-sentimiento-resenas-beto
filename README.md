# Proyecto final — Deep Learning

**Estudiante:** David Quintero Gallego

## Título

Clasificación del sentimiento en reseñas de negocios en español mediante *fine-tuning* de un Transformer (BETO) y evaluación de su generalización a reseñas reales de Google de negocios colombianos.

## Objetivo general

Desarrollar un clasificador de sentimiento de tres clases (negativo, neutral, positivo) para reseñas de negocios en español, mediante el ajuste fino (*fine-tuning*) del modelo preentrenado BETO, que alcance un F1 macro de al menos 0.70 en el conjunto de prueba, y cuantificar cuánto se mantiene o se degrada su desempeño al aplicarlo a reseñas reales de Google de negocios colombianos que no se usaron durante el entrenamiento.

## Objetivos específicos

1. Construir y preprocesar un corpus de entrenamiento a partir de un conjunto público de reseñas en español con calificación de 1 a 5 estrellas (por ejemplo, Rest-Mex de TripAdvisor o COAR). Las calificaciones se agruparán en tres clases de sentimiento y se dividirán de forma estratificada en entrenamiento, validación y prueba (70/15/15). Además, se construirá un conjunto externo de al menos 150 reseñas de Google Maps de negocios colombianos, etiquetadas manualmente.
2. Implementar y entrenar el clasificador ajustando BETO con la API Trainer de Hugging Face. Se seleccionarán los hiperparámetros (*learning rate*, número de épocas, longitud máxima) con base en las curvas de pérdida y de F1 de validación, y se comparará contra dos referencias: un baseline clásico (TF-IDF + regresión logística) y un Transformer multilingüe (DistilBERT multilingüe).
3. Evaluar el modelo en el conjunto de prueba y en el conjunto externo colombiano mediante accuracy, F1 macro, recall por clase y matriz de confusión. Se analizarán los errores y la degradación por cambio de dominio (país, plataforma y tipo de negocio) para identificar las fortalezas, las limitaciones y los casos en que falla.

## Resultados

| Modelo | F1 macro test | Accuracy test | Recall neutral | F1 macro test natural |
|---|---|---|---|---|
| TF-IDF + Regresión logística | 0.745 | 0.745 | 0.635 | 0.649 |
| DistilBERT multilingüe (lr 5e-5, 128 tokens) | 0.775 | 0.777 | 0.645 | 0.691 |
| **BETO** (lr 2e-5, 256 tokens) | 0.804 | 0.802 | 0.772 | 0.707 |

Test balanceado: 4.500 reseñas. Test natural: 10.000 reseñas con la distribución real (≈87% positivas). Meta del anteproyecto (F1 macro ≥ 0,70): **cumplida**. Conjunto colombiano (Google Maps, 166 reseñas de 19 negocios en 13 ciudades, etiquetado leyendo el texto): F1 macro BETO 0.854 (IC95 0.787–0.906), DistilBERT multilingüe 0.890 (0.826–0.941), TF-IDF 0.829 (0.760–0.886). Las diferencias entre modelos no son significativas con esta muestra (bootstrap); todos los errores de BETO son predecir “neutral”. Detalle en `resultados/externo_resumen.json`.

Guía de estudio interactiva: `guia/guia-defensa.html`.

## Estructura

```
src/
  config.py               rutas, semilla, regla estrellas -> clase, checkpoints de HF
  preparar_datos.py       Etapa 1: limpieza (ftfy), 3 clases, balanceo, split 70/15/15 + test natural
  baseline_tfidf.py       Etapa 2a: TF-IDF (1-2 gramas) + regresión logística
  entrenar_transformer.py Etapa 2b: fine-tuning de BETO / DistilBERT multilingüe con Trainer
  analisis_capas.py       Etapa 3b: sondas lineales y PCA del [CLS] en las 13 etapas internas
  evaluar_externo.py      Etapa 3c: evaluación en reseñas de Google Maps de Colombia
  demo.py                 demo en vivo: escribe una reseña y BETO la clasifica
  metricas.py, externo.py utilidades
data/externo/             plantilla e instrucciones del conjunto colombiano (etiquetado manual)
resultados/               métricas (JSON), historiales de entrenamiento, predicciones
guia/                     guía de estudio interactiva para la sustentación
```

## Demo en vivo

Con el modelo ya entrenado (queda en `modelos/_checkpoints/`), desde `src/`:

```bash
python demo.py --ejemplos     # frases preparadas y luego modo interactivo
python demo.py                # solo modo interactivo
python demo.py "texto"        # clasifica un único texto
```

Muestra la clase predicha y la probabilidad de cada clase. Funciona sin internet.

## Cómo reproducir

```bash
py -3.13 -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt
# Datos: Rest-Mex 2025 (CC-BY-4.0) desde Hugging Face
curl -L -o data/raw/Rest-Mex_2025_train.csv https://huggingface.co/datasets/vg055/Rest-Mex2025/resolve/main/Rest-Mex_2025_train.csv
cd src
python preparar_datos.py
python baseline_tfidf.py
bash ../correr_busqueda.sh                      # BETO con 3 learning rates + DistilBERT multilingüe
python analisis_capas.py --modelo_ft <mejor_checkpoint de BETO>
python evaluar_externo.py                       # cuando exista data/externo/resenas_google_colombia.csv
python ../guia/construir_guia.py
```

Hardware usado: NVIDIA RTX 4060 Laptop (8 GB), PyTorch 2.11 + CUDA 12.8, Transformers 5.

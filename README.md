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

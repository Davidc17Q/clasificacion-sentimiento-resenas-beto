# Conjunto externo: reseñas de Google Maps de negocios colombianos

Objetivo: medir cuánto se degrada el modelo fuera de su dominio (otro país, otra
plataforma, otros tipos de negocio). **Nunca se usa para entrenar.**

## Cómo construirlo (≥ 150 reseñas)

1. Copia `plantilla_resenas_google_colombia.csv` como `resenas_google_colombia.csv`.
2. Elige 10–15 negocios colombianos variados (restaurantes, cafés, hoteles, barberías,
   tiendas, talleres…) en 3+ ciudades. Abre sus reseñas en Google Maps y copia a mano
   10–15 reseñas de cada uno, **incluyendo las de 1–3 estrellas** (ordena por
   "Calificación más baja" para encontrarlas). Solo reseñas con texto.
3. Llena una fila por reseña:

| columna        | contenido                                                        |
|----------------|------------------------------------------------------------------|
| `texto`        | la reseña tal cual (entre comillas si tiene comas)               |
| `estrellas`    | 1–5, las que puso el autor                                       |
| `etiqueta`     | **tu** lectura del texto: `negativo`, `neutral` o `positivo`     |
| `negocio`      | nombre del negocio                                               |
| `ciudad`       | ciudad                                                           |
| `tipo_negocio` | restaurante, hotel, cafe, barberia, tienda, …                    |

Reglas de etiquetado manual (decide por el texto, no por las estrellas):
- **negativo**: predomina la queja o no recomendaría.
- **neutral**: mezcla equilibrada de bueno y malo, o descripción sin valoración clara.
- **positivo**: predomina el elogio o recomendaría.

Busca ~1/3 de cada clase si es posible (las negativas y neutrales son las que escasean).
No incluyas el nombre del autor (privacidad).

4. Ejecuta `python src/evaluar_externo.py` (usa el mejor checkpoint de BETO y de
   DistilBERT, y reentrena el TF-IDF en segundos) y luego `python guia/construir_guia.py`
   para actualizar la guía.

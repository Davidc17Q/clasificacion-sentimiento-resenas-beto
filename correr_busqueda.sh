#!/bin/bash
# Búsqueda de hiperparámetros (validación) + baseline multilingüe
set -e
cd "$(dirname "$0")/src"
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
for LR in 3e-5 2e-5 5e-5; do
  $PY entrenar_transformer.py --modelo beto --lr $LR --epocas 4 --max_len 128 > ../logs/beto_lr${LR}_len128.log 2>&1
  echo "LISTO beto lr=$LR"
done
$PY entrenar_transformer.py --modelo mdistilbert --lr 5e-5 --epocas 4 --max_len 128 > ../logs/mdistilbert.log 2>&1
echo "LISTO mdistilbert"

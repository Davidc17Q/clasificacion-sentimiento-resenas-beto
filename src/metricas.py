"""Métricas comunes a todos los modelos."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from config import ETIQUETAS


def calcular(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
        "por_clase": {
            ETIQUETAS[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                           "f1": round(float(f[i]), 4), "soporte": int(s[i])}
            for i in range(3)
        },
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }

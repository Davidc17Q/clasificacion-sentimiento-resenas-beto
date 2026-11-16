"""Etapa 2b — Fine-tuning de un Transformer (BETO o DistilBERT multilingüe) con la API Trainer.

Uso:
    python src/entrenar_transformer.py --modelo beto --lr 3e-5 --epocas 4 --max_len 128
    python src/entrenar_transformer.py --modelo mdistilbert --lr 5e-5

Cada corrida guarda en resultados/<nombre>/:
    metricas.json   (val / test / test_natural / externo)
    historial.json  (pérdida de entrenamiento y validación + F1 por época -> curvas)
    predicciones_test.csv
y el mejor checkpoint (por F1 macro de validación) en modelos/<nombre>/ si --guardar.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import metricas
from config import DATA_PROC, ID2LABEL, LABEL2ID, MODELOS, MODELOS_HF, RESULTADOS, SEMILLA
from externo import cargar_externo


def args_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", choices=list(MODELOS_HF), default="beto")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epocas", type=int, default=4)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--guardar", action="store_true", help="Guarda el mejor modelo en modelos/")
    p.add_argument("--nombre", default=None)
    return p.parse_args()


def main() -> None:
    a = args_cli()
    set_seed(SEMILLA)
    nombre = a.nombre or f"{a.modelo}_lr{a.lr:g}_len{a.max_len}"
    out_res = RESULTADOS / nombre
    out_res.mkdir(parents=True, exist_ok=True)
    checkpoints = MODELOS / "_checkpoints" / nombre

    checkpoint_hf = MODELOS_HF[a.modelo]
    tok = AutoTokenizer.from_pretrained(checkpoint_hf)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_hf, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def a_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["texto", "label"]].rename(columns={"label": "labels"}), preserve_index=False)
        return ds.map(lambda b: tok(b["texto"], truncation=True, max_length=a.max_len), batched=True,
                      remove_columns=["texto"])

    splits = {n: pd.read_csv(DATA_PROC / f"{n}.csv") for n in ["train", "val", "test", "test_natural"]}
    ds = {n: a_dataset(df) for n, df in splits.items()}

    # Cuántas reseñas superan max_len tokens (justifica la elección de longitud)
    largos = [len(x) for x in ds["train"]["input_ids"]]
    truncadas = float(np.mean([l >= a.max_len for l in largos]))

    def compute_metrics(pred):
        y_pred = pred.predictions.argmax(-1)
        m = metricas.calcular(pred.label_ids, y_pred)
        return {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}

    targs = TrainingArguments(
        output_dir=str(checkpoints),
        learning_rate=a.lr,
        num_train_epochs=a.epocas,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch * 2,
        weight_decay=0.01,
        warmup_steps=0.1,  # en Transformers 5 un float < 1 es fracción de los pasos (warmup del 10%)
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=SEMILLA,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=modelo,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    t0 = time.time()
    trainer.train()
    segundos = time.time() - t0

    salida = {
        "modelo": checkpoint_hf,
        "hiperparametros": {"lr": a.lr, "epocas_max": a.epocas, "max_len": a.max_len, "batch": a.batch,
                            "weight_decay": 0.01, "warmup_ratio": 0.1, "fp16": targs.fp16},
        "parametros_entrenables": sum(p.numel() for p in modelo.parameters() if p.requires_grad),
        "fraccion_truncada_train": round(truncadas, 4),
        "segundos_entrenamiento": round(segundos, 1),
        "mejor_checkpoint": trainer.state.best_model_checkpoint,
        "mejor_f1_val": trainer.state.best_metric,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    for n in ["val", "test", "test_natural"]:
        pred = trainer.predict(ds[n])
        y_pred = pred.predictions.argmax(-1)
        salida[n] = metricas.calcular(splits[n]["label"], y_pred)
        if n == "test":
            probs = torch.softmax(torch.tensor(pred.predictions, dtype=torch.float32), -1).numpy()
            df = splits[n].copy()
            df["pred"] = y_pred
            df[["p_neg", "p_neu", "p_pos"]] = probs
            df.to_csv(out_res / "predicciones_test.csv", index=False)

    ext = cargar_externo()
    if ext is not None:
        ds_ext = a_dataset(ext)
        pred = trainer.predict(ds_ext)
        salida["externo_colombia"] = metricas.calcular(ext["label"], pred.predictions.argmax(-1))

    (out_res / "metricas.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_res / "historial.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")

    if a.guardar:
        destino = MODELOS / nombre
        trainer.save_model(str(destino))
        tok.save_pretrained(str(destino))

    print(json.dumps({k: salida[k]["f1_macro"] if isinstance(salida[k], dict) and "f1_macro" in salida[k]
                      else salida[k] for k in salida}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

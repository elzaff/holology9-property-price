"""Retrain the complete HOLOGY v5 recipe on Modal and write ``submissionv5.csv``.

Run from this directory after placing ``data/train.csv`` and ``data/test.csv`` here:

    modal run solutions.py

The script launches the three 5-fold OOF components used by v3, then two additional
full-train DeBERTa seeds used by v4/v5. Final v5 prediction is:

    0.78 * mean(3 DeBERTa runs) + 0.22 * mean(2 ModernBERT runs)

The blend and log-space recalibration constants are frozen from the OOF-selected v5
recipe. No account identifiers, credentials, or external data are stored here.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import modal


APP_NAME = "holo-property-v5"
DATA_DIR = os.environ.get("HOLO_DATA_DIR", "data")
GPU = os.environ.get("HOLO_GPU", "A100-40GB")
N_SPLITS = 5
FOLD_SEED = 42
EPOCHS = 6
BATCH_SIZE = 4

# Frozen v5 recipe. These are applied after fresh models are retrained.
V5_DEBERTA_WEIGHT = 0.78
V5_MODERNBERT_WEIGHT = 0.22
V5_RECALIBRATION_A = 0.3260
V5_RECALIBRATION_B = 0.9766

RUNS = (
    {"tag": "debL", "model": "microsoft/deberta-v3-large", "maxlen": 512,
     "lr": 8e-6, "seed": 0, "full_train": False},
    {"tag": "mbertL", "model": "answerdotai/ModernBERT-large", "maxlen": 1024,
     "lr": 1e-5, "seed": 0, "full_train": False},
    {"tag": "mbertL2", "model": "answerdotai/ModernBERT-large", "maxlen": 1024,
     "lr": 1e-5, "seed": 1, "full_train": False},
    {"tag": "debL_f1", "model": "microsoft/deberta-v3-large", "maxlen": 512,
     "lr": 8e-6, "seed": 1, "full_train": True},
    {"tag": "debL_f2", "model": "microsoft/deberta-v3-large", "maxlen": 512,
     "lr": 8e-6, "seed": 2, "full_train": True},
)

HF_CACHE = modal.Volume.from_name("holo-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.48.0",
        "scikit-learn==1.5.2",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "sentencepiece",
        "protobuf",
        "tiktoken",
        "scipy==1.14.1",
    )
    .env({"HF_HOME": "/cache", "TRANSFORMERS_CACHE": "/cache"})
    .add_local_dir(DATA_DIR, "/data")
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 60 * 60,
    volumes={"/cache": HF_CACHE},
    max_containers=15,
)
def train_one(job: dict):
    """Train one model/fold and return log-space validation and submission predictions."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_cosine_schedule_with_warmup,
    )

    tag = job["tag"]
    fold = int(job["fold"])
    model_name = job["model"]
    maxlen = int(job["maxlen"])
    lr = float(job["lr"])
    seed = int(job["seed"])

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train = pd.read_csv("/data/train.csv")
    submit = pd.read_csv("/data/test.csv")
    train["text"] = train["text"].fillna("")
    submit["text"] = submit["text"].fillna("")

    price = train["listPrice"].to_numpy(float)
    y_log = np.log1p(price).astype("float32")
    mu, sd = float(y_log.mean()), float(y_log.std())
    y_scaled = (y_log - mu) / sd

    if fold < 0:
        train_idx = np.arange(len(train))
        valid_idx = None
    else:
        splits = list(KFold(N_SPLITS, shuffle=True, random_state=FOLD_SEED)
                      .split(np.arange(len(train))))
        train_idx, valid_idx = splits[fold]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorWithPadding(tokenizer)

    class Listings(Dataset):
        def __init__(self, texts, targets=None):
            self.texts = list(texts)
            self.targets = targets

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            item = tokenizer(self.texts[index], truncation=True, max_length=maxlen)
            if self.targets is not None:
                item["labels"] = float(self.targets[index])
            return item

    generator = torch.Generator()
    generator.manual_seed(seed + max(fold, 0))

    loader_train = DataLoader(
        Listings(train["text"].values[train_idx], y_scaled[train_idx]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=2,
        drop_last=True,
    )
    loader_submit = DataLoader(
        Listings(submit["text"].values),
        batch_size=BATCH_SIZE * 2,
        collate_fn=collator,
        num_workers=2,
    )
    loader_valid = None if valid_idx is None else DataLoader(
        Listings(train["text"].values[valid_idx]),
        batch_size=BATCH_SIZE * 2,
        collate_fn=collator,
        num_workers=2,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(loader_train) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    loss_fn = nn.L1Loss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in loader_train:
            labels = batch.pop("labels").cuda().float()
            batch = {key: value.cuda() for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                output = model(**batch).logits.squeeze(-1).float()
                loss = loss_fn(output, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        print(f"[{tag}] fold={fold} epoch={epoch} loss={total_loss / len(loader_train):.4f}", flush=True)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        predictions = []
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.cuda() for key, value in batch.items()}
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                predictions.append(model(**batch).logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(predictions) * sd + mu

    valid_log = None if loader_valid is None else predict(loader_valid)
    submit_log = predict(loader_submit)
    valid_indices = None if valid_idx is None else np.asarray(valid_idx, dtype=np.int64)
    if valid_log is not None:
        valid_mae = float(np.abs(np.expm1(valid_log) - price[valid_indices]).mean())
        print(f"[{tag}] fold={fold} validation MAE={valid_mae:,.0f}", flush=True)

    return {
        "tag": tag,
        "fold": fold,
        "valid_indices": valid_indices,
        "valid_log": valid_log,
        "submit_log": submit_log,
    }


def _mae_from_log(price, prediction_log):
    import numpy as np

    return float(np.abs(np.expm1(prediction_log) - price).mean())


@app.local_entrypoint()
def main(
    out: str = "submissionv5.csv",
    artifacts: str = "modal_v5_artifacts",
    dry_run: bool = False,
):
    """Launch all jobs, blend their outputs, and write the v5 submission locally."""
    import numpy as np
    import pandas as pd

    train = pd.read_csv(Path(DATA_DIR) / "train.csv")
    submit = pd.read_csv(Path(DATA_DIR) / "test.csv")
    price = train["listPrice"].to_numpy(float)

    jobs = []
    for run in RUNS:
        folds = [-1] if run["full_train"] else list(range(N_SPLITS))
        jobs.extend({**run, "fold": fold} for fold in folds)

    if dry_run:
        print(f"GPU={GPU}; jobs={len(jobs)}; output={out}")
        for job in jobs:
            print(job)
        return

    print(f"Launching {len(jobs)} Modal jobs on {GPU}...")
    results = {}
    for result in train_one.map(jobs):
        results[(result["tag"], result["fold"])] = result
        print(f"completed {result['tag']} fold={result['fold']}", flush=True)

    artifact_dir = Path(artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    oof = {}
    submit_log = {}

    for run in RUNS:
        tag = run["tag"]
        if run["full_train"]:
            submit_log[tag] = results[(tag, -1)]["submit_log"]
            np.save(artifact_dir / f"test_{tag}.npy", submit_log[tag])
            continue

        oof[tag] = np.zeros(len(train), dtype=np.float32)
        fold_predictions = []
        for fold in range(N_SPLITS):
            result = results[(tag, fold)]
            oof[tag][result["valid_indices"]] = result["valid_log"]
            fold_predictions.append(result["submit_log"])
        submit_log[tag] = np.mean(fold_predictions, axis=0)
        np.save(artifact_dir / f"oof_{tag}.npy", oof[tag])
        np.save(artifact_dir / f"test_{tag}.npy", submit_log[tag])
        print(f"{tag} OOF MAE={_mae_from_log(price, oof[tag]):,.1f}")

    deb3 = np.mean([submit_log["debL"], submit_log["debL_f1"], submit_log["debL_f2"]], axis=0)
    mb2 = (submit_log["mbertL"] + submit_log["mbertL2"]) / 2
    blend_log = V5_DEBERTA_WEIGHT * deb3 + V5_MODERNBERT_WEIGHT * mb2
    final_prediction = np.maximum(
        0.0,
        np.expm1(V5_RECALIBRATION_A + V5_RECALIBRATION_B * blend_log),
    )

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": submit["id"], "listPrice": final_prediction})
    frame.to_csv(output_path, index=False)
    if output_path.name == "submissionv5.csv":
        frame.to_csv(output_path.with_name("submission_v5.csv"), index=False)

    np.save(artifact_dir / "v5_blend_log.npy", blend_log)
    print(f"wrote {output_path} ({len(frame)} rows)")
    print(f"artifacts: {artifact_dir.resolve()}")

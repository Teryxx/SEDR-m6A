import argparse
import csv
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset


DEFAULT_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_PACKAGE_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(DEFAULT_PACKAGE_ROOT / "code"))

from structure_fusion.model_ernie_token_dpf import ATTN_PADDED_LEN, TOKEN_LEN  # noqa: E402
from structure_fusion.model_splicebert_motif import MotifAwareDPFClassifier, center_drach_to_id  # noqa: E402


METRIC_FIELDS = ["ACC", "MCC", "PRE", "REC", "F1"]
ERNIE_HEAD6_CHANNEL_INDEX = 149
ERNIE_FULL_CACHE_SHAPE = (156, ATTN_PADDED_LEN, ATTN_PADDED_LEN)
ERNIE_SLIM_CACHE_SHAPE = (1, ATTN_PADDED_LEN, ATTN_PADDED_LEN)


def compute_metrics(labels, preds):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    return {
        "ACC": float(accuracy_score(labels, preds)),
        "MCC": float(matthews_corrcoef(labels, preds)),
        "PRE": float(precision_score(labels, preds, zero_division=0)),
        "REC": float(recall_score(labels, preds, zero_division=0)),
        "F1": float(f1_score(labels, preds, zero_division=0)),
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def infer_package_root(manifest_path: Path) -> Path:
    manifest_path = manifest_path.resolve()
    if manifest_path.name != "final11_manifest.json":
        return manifest_path.parent.parent
    if manifest_path.parent.name == "checkpoint":
        return manifest_path.parent.parent
    return manifest_path.parent


def _patch_transformers_for_multimolecule():
    import transformers

    def _noop_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    if not hasattr(transformers.utils.generic, "check_model_inputs"):
        transformers.utils.generic.check_model_inputs = _noop_decorator
    if not hasattr(transformers.utils.generic, "can_return_tuple"):
        transformers.utils.generic.can_return_tuple = _noop_decorator


def _window_sequence(sequence: str, size: int) -> str:
    sequence = str(sequence).strip().upper().replace("U", "T")
    size = min(int(size), len(sequence))
    mid = len(sequence) // 2
    left_size = size // 2
    right_size = size - left_size
    return sequence[mid - left_size: mid + right_size]


def _load_split_texts_labels(data_dir: Path, dataset: str, split_name: str, window_size: int, max_samples=None):
    path = data_dir / f"{dataset}_{split_name}.tsv"
    texts = []
    labels = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "label" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"{path} must contain 'label' and 'text' columns")
        for row in reader:
            seq = _window_sequence(row["text"], window_size)
            if len(seq) != TOKEN_LEN:
                raise ValueError(f"{path} contains a {len(seq)}nt window; expected {TOKEN_LEN}nt")
            texts.append(seq)
            labels.append(int(float(row["label"])))
            if max_samples is not None and len(texts) >= int(max_samples):
                break
    return texts, np.asarray(labels, dtype=np.int64)


def _load_metadata(path: Path):
    if not path.exists():
        return None
    metadata = torch.load(path, map_location="cpu")
    return metadata if isinstance(metadata, dict) else None


def _validate_ernie_cache_shape(path: Path, ernie_format: str):
    ernie = np.load(path, mmap_mode="r")
    shape = tuple(ernie.shape)
    if ernie_format == "head6_v1":
        if np.dtype(ernie.dtype) != np.dtype(np.float32):
            raise ValueError(f"Expected slim ERNIE cache dtype float32, got {ernie.dtype}")
        if ernie.ndim != 4 or shape[1:] != ERNIE_SLIM_CACHE_SHAPE:
            raise ValueError(f"Expected slim ERNIE cache shape (N,1,203,203), got {shape}")
        return
    if ernie_format == "full156_v1":
        if np.dtype(ernie.dtype) != np.dtype(np.float16):
            raise ValueError(f"Expected full ERNIE cache dtype float16, got {ernie.dtype}")
        if ernie.ndim != 4 or shape[1:] != ERNIE_FULL_CACHE_SHAPE:
            raise ValueError(f"Expected full ERNIE cache shape (N,156,203,203), got {shape}")
        return
    raise ValueError(f"Unsupported ERNIE cache format: {ernie_format}")


def _load_ernie_cache(package_root: Path, dataset: str, split_name: str, labels):
    cache_dir = package_root / "data" / "structure_fusion" / "cache" / f"{dataset}_{split_name}"
    metadata = _load_metadata(cache_dir / "metadata.pt")
    ernie_format = metadata.get("ernie_format") if metadata else "head6_v1"
    ernie_name = "ernie_head6.npy" if ernie_format == "head6_v1" else "ernie_attn.npy"
    ernie_path = cache_dir / ernie_name
    if not ernie_path.exists():
        raise FileNotFoundError(ernie_path)
    _validate_ernie_cache_shape(ernie_path, ernie_format)

    expected_labels = np.asarray(labels, dtype=np.int64)
    labels_path = cache_dir / "labels.npy"
    if labels_path.exists():
        cached_labels = np.asarray(np.load(labels_path), dtype=np.int64)[: len(expected_labels)]
        if cached_labels.shape[0] != expected_labels.shape[0] or not np.array_equal(cached_labels, expected_labels):
            raise ValueError(f"{dataset}_{split_name} label mismatch between TSV and ERNIE cache")
    if metadata and "labels" in metadata:
        metadata_labels = np.asarray(metadata["labels"], dtype=np.int64)[: len(expected_labels)]
        if metadata_labels.shape[0] != expected_labels.shape[0] or not np.array_equal(metadata_labels, expected_labels):
            raise ValueError(f"{dataset}_{split_name} label mismatch between TSV and ERNIE metadata")

    ernie = np.load(ernie_path, mmap_mode="r")
    if ernie.shape[0] < len(expected_labels):
        raise ValueError(f"ERNIE cache has {ernie.shape[0]} samples but TSV requires {len(expected_labels)}")
    return {
        "cache_dir": str(cache_dir),
        "ernie_path": str(ernie_path),
        "ernie_format": ernie_format,
        "num_samples": len(expected_labels),
        "labels": expected_labels,
    }


class LiveSpliceBERTMotifHead6Dataset(Dataset):
    def __init__(self, texts, labels, motif_ids, ernie_cache, indices=None):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.motif_ids = np.asarray(motif_ids, dtype=np.int64)
        self.indices = np.asarray(indices if indices is not None else np.arange(len(self.labels)), dtype=np.int64)
        self.ernie = np.load(ernie_cache["ernie_path"], mmap_mode="r")
        self.ernie_format = ernie_cache["ernie_format"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        ernie = torch.from_numpy(
            np.array(
                self.ernie[idx],
                dtype=np.float32 if self.ernie_format == "head6_v1" else np.float16,
                copy=True,
            )
        )
        if self.ernie_format == "full156_v1":
            ernie = ernie[ERNIE_HEAD6_CHANNEL_INDEX:ERNIE_HEAD6_CHANNEL_INDEX + 1].to(dtype=torch.float32)
        elif ernie.shape[0] != 1:
            raise ValueError(f"Expected slim ERNIE cache item to have 1 channel, got {tuple(ernie.shape)}")
        return {
            "splicebert_sequence": str(self.texts[idx]),
            "ernie_attn": ernie,
            "motif_ids": torch.tensor(int(self.motif_ids[idx]), dtype=torch.long),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_loader(dataset, batch_size: int, num_workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


class SpliceBERTMotifInferenceModel(nn.Module):
    def __init__(self, motif_model: nn.Module, model_name_or_path: str, device: torch.device):
        super().__init__()
        _patch_transformers_for_multimolecule()
        from multimolecule import RnaTokenizer
        from transformers import AutoModel

        self.motif_model = motif_model
        self.tokenizer = RnaTokenizer.from_pretrained(model_name_or_path)
        self.splicebert_model = AutoModel.from_pretrained(model_name_or_path).to(device)
        for param in self.splicebert_model.parameters():
            param.requires_grad = False
        self.hidden_dim = int(getattr(self.splicebert_model.config, "hidden_size", 512))
        self.device = device

    def _extract_tokens(self, seqs):
        seqs = [str(seq).strip().upper() for seq in seqs]
        for index, seq in enumerate(seqs):
            if len(seq) != TOKEN_LEN:
                raise ValueError(f"Sequence at index {index} has length {len(seq)}, expected {TOKEN_LEN}")
        encoded = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.splicebert_model(**encoded)
        tokens = outputs.last_hidden_state[:, 1:-1, :]
        if tokens.shape[1] != TOKEN_LEN:
            raise ValueError(f"Expected SpliceBERT token length {TOKEN_LEN}, got {tokens.shape[1]}")
        return tokens

    def forward(self, batch):
        if "splicebert_sequences" in batch:
            batch = dict(batch)
            batch["ernie_tokens"] = self._extract_tokens(batch.pop("splicebert_sequences"))
        return self.motif_model(batch)


def _make_motif_model(config, device):
    if config.get("experiment", "motif_aware") != "motif_aware":
        raise ValueError("This offline runner expects motif_aware final checkpoints")
    signature = inspect.signature(MotifAwareDPFClassifier.__init__)
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name == "track_gate_regularization":
            kwargs[name] = float(config.get("gate_balance_l2", 0.0)) > 0.0
        elif name in config:
            kwargs[name] = config[name]
        elif parameter.default is not inspect._empty:
            kwargs[name] = parameter.default
    if kwargs.get("gated_residual_scale") is None:
        kwargs["gated_residual_scale"] = kwargs.get("gate_residual_scale", 1.0)
    return MotifAwareDPFClassifier(**kwargs).to(device)


def _runtime_config(package_root: Path, config_path: Path, batch_size: int, num_workers: int):
    config = _load_json(config_path)
    config["data_dir"] = str(package_root / "data" / "preprocessed_dataset")
    config["ernie_cache_dir"] = str(package_root / "data" / "structure_fusion" / "cache")
    config["splicebert_model"] = str(package_root / "checkpoint" / "pretrained_weights" / "splicebert-human.510")
    config["batch_size"] = int(batch_size)
    config["num_workers"] = int(num_workers)
    config.setdefault("window_size", TOKEN_LEN)
    config.setdefault("max_samples", None)
    config.setdefault("seed", 42)
    return config


def _load_model(config, checkpoint_path: Path, device):
    motif_model = _make_motif_model(config, device)
    model = SpliceBERTMotifInferenceModel(
        motif_model=motif_model,
        model_name_or_path=config["splicebert_model"],
        device=device,
    ).to(device)
    if int(config.get("hidden_dim", model.hidden_dim)) != int(model.hidden_dim):
        raise ValueError(f"hidden_dim={config.get('hidden_dim')} does not match SpliceBERT hidden_dim={model.hidden_dim}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def _prepare_batch(batch, device):
    labels = batch["labels"].to(device, non_blocking=True)
    model_batch = {
        "splicebert_sequences": list(batch["splicebert_sequence"]),
        "motif_ids": batch["motif_ids"].to(device=device, dtype=torch.long, non_blocking=True),
        "ernie_attn": batch["ernie_attn"].to(device=device, dtype=torch.float32, non_blocking=True),
    }
    return labels, model_batch


@torch.no_grad()
def _predict_with_model(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    for batch in loader:
        labels, model_batch = _prepare_batch(batch, device)
        logits, _ = model(model_batch)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return np.asarray(all_labels, dtype=np.int64), np.asarray(all_preds, dtype=np.int64)


def evaluate_one(package_root: Path, entry, batch_size: int, num_workers: int, device):
    config_path = package_root / entry["config"]
    checkpoint_path = package_root / entry["checkpoint"]
    config = _runtime_config(package_root, config_path, batch_size=batch_size, num_workers=num_workers)
    texts, labels = _load_split_texts_labels(
        Path(config["data_dir"]),
        entry["dataset"],
        "test",
        window_size=int(config["window_size"]),
        max_samples=config.get("max_samples"),
    )
    motif_ids = np.asarray([center_drach_to_id(seq) for seq in texts], dtype=np.int64)
    ernie_cache = _load_ernie_cache(package_root, entry["dataset"], "test", labels)
    indices = np.arange(len(labels))
    dataset = LiveSpliceBERTMotifHead6Dataset(texts, labels, motif_ids, ernie_cache, indices=indices)
    loader = _build_loader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        seed=int(config["seed"]),
    )
    model = _load_model(config, checkpoint_path, device)
    labels, preds = _predict_with_model(model, loader, device)
    metrics = compute_metrics(labels, preds)
    return {"dataset": entry["dataset"], **metrics}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the offline final11 SEDR-m6A checkpoints.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PACKAGE_ROOT / "checkpoint" / "final11_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", type=str, default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    package_root = infer_package_root(args.manifest)
    if args.output is None:
        args.output = package_root / "test_results.csv"
    wanted = {x.strip() for x in args.datasets.split(",") if x.strip()}
    manifest = _load_json(args.manifest)
    if wanted:
        manifest = [entry for entry in manifest if entry["dataset"] in wanted]
    device = torch.device(args.device)
    rows = []
    for entry in manifest:
        row = evaluate_one(package_root, entry, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        rows.append(row)
        print(
            "{dataset}\tACC={ACC:.10f}\tMCC={MCC:.10f}\tPRE={PRE:.10f}\tREC={REC:.10f}\tF1={F1:.10f}".format(
                **row
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", *METRIC_FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

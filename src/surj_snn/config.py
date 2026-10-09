from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Config:
    name: str = "main"
    dataset: str = "FashionMNIST"
    data_dir: str = "data"
    results_dir: str = "runs/main"
    train_size: int = 54000
    val_size: int = 6000
    split_seed: int = 2026
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64])
    beta: float = 0.9
    threshold: float = 1.0
    surrogate_alpha: float = 2.0
    logit_scale: float = 10.0
    timesteps: list[int] = field(default_factory=lambda: [8, 16, 32])
    penalties: list[float] = field(default_factory=lambda: [0.0, 0.1, 1.0])
    threshold_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.25, 1.5])
    seeds: list[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])
    validation_encoding_seeds: list[int] = field(default_factory=lambda: [101, 102, 103])
    test_encoding_seeds: list[int] = field(default_factory=lambda: [201, 202, 203])
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 15
    learning_rate: float = 0.001
    accuracy_tolerance_pp: float = 1.0
    primary_timestep: int = 16
    primary_penalty: float = 0.1
    device: str = "cpu"
    threads: int = 4
    deterministic: bool = True
    include_ann: bool = True

    def validate(self):
        if self.dataset not in {"FashionMNIST", "MNIST", "synthetic"}:
            raise ValueError("dataset must be FashionMNIST, MNIST, or synthetic")
        for key in ("train_size", "val_size", "batch_size", "eval_batch_size", "epochs", "threads"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.train_size % 10 or self.val_size % 10:
            raise ValueError("Stratified subset sizes must be multiples of ten")
        if self.train_size + self.val_size > (500 if self.dataset == "synthetic" else 60000):
            raise ValueError("Requested subsets exceed the available training partition")
        if len(self.hidden_sizes) != 2 or any(n <= 0 for n in self.hidden_sizes):
            raise ValueError("Exactly two positive hidden widths are required")
        for key in ("timesteps", "penalties", "threshold_multipliers", "seeds", "validation_encoding_seeds", "test_encoding_seeds"):
            values = getattr(self, key)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{key} must be a nonempty list of unique values")
        if any(not isinstance(t, int) or t <= 0 for t in self.timesteps):
            raise ValueError("Timesteps must be positive integers")
        if any(not math.isfinite(p) or p < 0 for p in self.penalties) or 0 not in self.penalties:
            raise ValueError("Penalties must be finite, nonnegative, and include zero")
        if any(not math.isfinite(a) or a < 1 for a in self.threshold_multipliers) or 1 not in self.threshold_multipliers:
            raise ValueError("Threshold multipliers must include one and be at least one")
        if not 0 <= self.beta <= 1 or self.threshold <= 0:
            raise ValueError("Invalid LIF dynamics")
        if self.learning_rate <= 0 or self.logit_scale <= 0 or self.surrogate_alpha <= 0:
            raise ValueError("Learning rate, logit scale, and surrogate alpha must be positive")
        if self.primary_timestep not in self.timesteps or self.primary_penalty not in self.penalties:
            raise ValueError("Primary contrast must occur in the declared grid")
        if self.primary_penalty <= 0 or self.accuracy_tolerance_pp < 0:
            raise ValueError("Primary penalty must be positive; tolerance must be nonnegative")
        if set(self.validation_encoding_seeds) & set(self.test_encoding_seeds):
            raise ValueError("Validation and test encoding seeds must be disjoint")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        return self

    def as_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    @property
    def output(self):
        return ROOT / self.results_dir

def load_config(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return Config(**json.load(stream)).validate()

def token(value):
    return format(float(value), ".10g").replace(".", "p").replace("-", "m")

def run_name(kind, steps, penalty, seed):
    return f"ann_seed{seed}" if kind == "ann" else f"snn_T{steps}_L{token(penalty)}_seed{seed}"

def conditions(cfg):
    for steps in cfg.timesteps:
        for penalty in cfg.penalties:
            for seed in cfg.seeds:
                yield "snn", steps, penalty, seed
    if cfg.include_ann:
        for seed in cfg.seeds:
            yield "ann", 1, 0.0, seed


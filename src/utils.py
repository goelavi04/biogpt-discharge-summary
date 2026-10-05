"""
Shared helpers used by every other script in src/: config loading, seeding,
device selection, tokenizer loading, and small I/O utilities.

WHY this file exists: data_prep.py, baseline_demo.py, train.py, generate.py,
and evaluate.py all need the *same* seed, the *same* tokenizer settings, and
the *same* config.yaml parsing. Duplicating that logic five times is how you
end up with five scripts that are subtly out of sync (e.g. one seeds numpy
but not torch, another loads BioGPT's tokenizer without the pad-token fix
below). Centralizing it here means a fix or change only has to happen once.
"""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

# Resolve paths relative to the project root (the directory containing this
# src/ folder), not the current working directory. This matters because
# Colab notebooks, pytest, and `python src/train.py` from different cwds
# would otherwise all resolve config.yaml / outputs/ differently.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_logger(name: str) -> logging.Logger:
    """One consistent logging format across all scripts."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def load_config(config_path: str | os.PathLike = "config.yaml") -> dict:
    """
    Load config.yaml and resolve every entry under `paths:` (plus the two
    dataset raw_path entries) to absolute paths anchored at PROJECT_ROOT.

    WHY resolve here instead of in each script: it's the one place that
    needs to know "paths in config.yaml are relative to the repo root", so
    every downstream script can just do `cfg["paths"]["outputs_dir"]` and
    get something that works regardless of where the script was launched
    from.

    The BIOGPT_CONFIG env var, if set, overrides config_path -- e.g. to point
    every script at a tiny smoke-test config without editing config.yaml.
    """
    config_path = Path(os.environ.get("BIOGPT_CONFIG", config_path))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str(PROJECT_ROOT / value)

    for mode in ("mimic_bhc", "mtsamples"):
        raw_path = cfg["dataset"][mode].get("raw_path")
        if raw_path:
            cfg["dataset"][mode]["raw_path"] = str(PROJECT_ROOT / raw_path)

    return cfg


def set_seed(seed: int) -> None:
    """
    Seed every source of randomness we actually use: Python's `random`,
    numpy, and torch (CPU + all CUDA devices). Also sets PYTHONHASHSEED so
    hash-based ordering (e.g. in some dict/set iteration edge cases) is
    reproducible across processes.

    WHY this matters here specifically: the project does an 80/10/10 split
    (data_prep.py), LoRA weight init and dropout (train.py), and sampling
    decoding (baseline_demo.py, generate.py) — all three are sources of
    nondeterminism that would otherwise make "rerun this and get the same
    numbers for the model card" false.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """cuda if available (Colab T4), else cpu (local inference-only demos)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_biogpt_tokenizer(model_name: str):
    """
    Load BioGPT's tokenizer with the one fix it needs for our use case:
    BioGPT ships without a dedicated pad token (it was pretrained purely for
    causal LM, where padding wasn't needed). We batch variable-length
    examples during training, which requires padding, so we fall back to
    using EOS as PAD. This is the standard fix for GPT-style tokenizers and
    is safe here because we mask padded positions out of the loss anyway
    (see train.py's data collator).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt_ids(tokenizer, input_text: str, cfg: dict) -> list[int]:
    """
    Build the token-id prompt (template prefix + possibly-truncated input +
    template middle), with NO target tokens. Shared by train.py (which
    appends target ids and masks the prompt portion out of the loss) and
    generate.py (where this is exactly what gets fed to model.generate()),
    so the truncation behavior used at training time and at inference time
    can never silently drift apart between the two scripts.
    """
    prompt_prefix, prompt_middle = cfg["prompt_template"].split("{input}")
    max_input_tokens = cfg["dataset"]["max_input_tokens"]

    prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    middle_ids = tokenizer.encode(prompt_middle, add_special_tokens=False)
    input_ids_body = tokenizer.encode(input_text, add_special_tokens=False)[:max_input_tokens]
    return prefix_ids + input_ids_body + middle_ids


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str | os.PathLike, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

"""
Load, clean, split, and tokenize-length-analyze the (clinical note -> brief
hospital course) dataset, in whichever of the two supported modes
config.yaml selects.

Output of running this script:
  data/processed/{train,val,test}.jsonl   -- the actual (input, target) pairs
  outputs/dataset_stats.json              -- counts, token-length stats, truncation rate
  outputs/figures/token_length_histogram.png

WHY two modes: MIMIC-IV-Ext-BHC is the "real" dataset for this project, but
it's PhysioNet credentialed-access data that can't be downloaded
automatically or redistributed -- you have to get it yourself and point
config.yaml at it. MTSamples is a public, no-credentialing-required stand-in
so the rest of the pipeline (and this teaching exercise) can run end-to-end
without PhysioNet access. The two loaders below both funnel into the same
(id, input, target) schema so everything downstream (train/generate/evaluate)
is dataset-mode-agnostic.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    ensure_dir,
    get_logger,
    load_biogpt_tokenizer,
    load_config,
    save_json,
    set_seed,
)

logger = get_logger("data_prep")


# ============================================================================
# Cleaning
# ============================================================================

# MIMIC (and some MTSamples exports) replace de-identified spans -- names,
# dates, facility names -- with runs of underscores, e.g. "Pt seen on ___ by
# Dr. ___." These are meaningless placeholders for a generation model if left
# in, so we collapse them to a single neutral token.
DEID_PATTERN = re.compile(r"_{2,}")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = DEID_PATTERN.sub("[REDACTED]", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = BLANK_LINES_PATTERN.sub("\n\n", text)
    return text.strip()


# ============================================================================
# MTSamples: regex-based section splitting
# ============================================================================

# Matches section headers at the start of a line, e.g. "HOSPITAL COURSE:",
# "HISTORY OF PRESENT ILLNESS:", "DISCHARGE DIAGNOSIS:". This is a heuristic
# -- MTSamples transcriptions are dictation transcripts, not structured EHR
# fields, so header formatting is inconsistent. We accept that and instead
# rely on dropping rows where the heuristic fails (see split_hospital_course)
# rather than trying to handle every edge case.
HEADER_LINE_RE = re.compile(r"(?m)^[ \t]*([A-Z][A-Z0-9 /\-]{2,45}):[ \t]*")

# Candidate header text (after matching HEADER_LINE_RE) that we treat as the
# "brief hospital course" target section. Order doesn't matter; we just need
# one of these substrings to appear in the matched header.
TARGET_HEADER_SUBSTRINGS = [
    "BRIEF HOSPITAL COURSE",
    "HOSPITAL COURSE",
    "SUMMARY OF HOSPITAL COURSE",
    "COURSE IN THE HOSPITAL",
    "HOSPITAL COURSE AND TREATMENT",
]

MIN_TARGET_CHARS = 40  # guards against matching an empty/near-empty section

# The Kaggle MTSamples export (and its HF mirrors) flattened every line break
# into a comma, e.g. "...(TAH).,COMPLICATIONS:,  None.,HOSPITAL COURSE: , The
# patient...". With no newlines, HEADER_LINE_RE (which anchors on line start)
# can never match, so we put the line breaks back first: ".," between
# sentences/list items, and "," directly before an ALL-CAPS "HEADER:".
FLATTENED_LINE_BREAK_RE = re.compile(r"\.,(?=\S)")
FLATTENED_HEADER_BREAK_RE = re.compile(r",\s*(?=[A-Z][A-Z0-9 /\-]{2,45}:)")


def restore_line_breaks(text: str) -> str:
    if not isinstance(text, str) or "\n" in text:
        return text
    text = FLATTENED_LINE_BREAK_RE.sub(".\n", text)
    return FLATTENED_HEADER_BREAK_RE.sub("\n", text)


def split_hospital_course(text: str) -> tuple[str, str] | tuple[None, None]:
    """
    Find a hospital-course-like section header, split the text into
    (input_without_that_section, target_section_content). Returns (None,
    None) if no matching header is found, or if the matched section is
    suspiciously short (likely a mis-match).
    """
    headers = list(HEADER_LINE_RE.finditer(text))
    if not headers:
        return None, None

    target_idx = None
    for i, m in enumerate(headers):
        header_text = m.group(1).upper()
        if any(sub in header_text for sub in TARGET_HEADER_SUBSTRINGS):
            target_idx = i
            break

    if target_idx is None:
        return None, None

    target_match = headers[target_idx]
    content_start = target_match.end()
    content_end = headers[target_idx + 1].start() if target_idx + 1 < len(headers) else len(text)
    # strip(" ,\n"): flattened exports leave a stray comma after the header ("HOSPITAL COURSE: , The ...").
    target_content = text[content_start:content_end].strip(" ,\n")

    if len(target_content) < MIN_TARGET_CHARS:
        return None, None

    # Input = everything else: text before the matched header + text from
    # the next header onward. This keeps the rest of the note (HPI, meds,
    # etc.) as model input while removing the exact section it has to predict.
    before = text[: target_match.start()]
    after = text[content_end:]
    input_text = (before + "\n" + after).strip()

    if len(input_text) < MIN_TARGET_CHARS:
        return None, None

    return input_text, target_content


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_mtsamples(cfg: dict) -> pd.DataFrame:
    ms_cfg = cfg["dataset"]["mtsamples"]
    raw_path = Path(ms_cfg["raw_path"])

    if raw_path.exists():
        logger.info(f"Loading MTSamples from local CSV: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        hf_name = ms_cfg.get("hf_dataset_name")
        logger.info(
            f"Local MTSamples CSV not found at {raw_path}; "
            f"trying Hugging Face Hub mirror '{hf_name}'"
        )
        from datasets import load_dataset

        ds = load_dataset(hf_name, split="train")
        df = ds.to_pandas()

    specialty_col = _find_column(df, ["medical_specialty", "specialty"])
    transcription_col = _find_column(df, ["transcription", "text", "note"])
    if transcription_col is None:
        raise ValueError(
            f"Could not find a transcription/text column in MTSamples data. "
            f"Available columns: {list(df.columns)}"
        )

    if not ms_cfg.get("keep_specialties"):
        logger.info("keep_specialties is null: using all specialties (regex split does the filtering).")
    elif specialty_col is not None:
        keep = [s.lower() for s in ms_cfg["keep_specialties"]]
        before = len(df)
        # .str.strip(): specialty values in the export carry a leading space (" Discharge Summary").
        df = df[df[specialty_col].astype(str).str.strip().str.lower().isin(keep)]
        logger.info(f"Filtered to specialties {ms_cfg['keep_specialties']}: {before} -> {len(df)} rows")
    else:
        logger.warning("No specialty column found; using all rows (will rely on regex split to filter).")

    df = df.dropna(subset=[transcription_col]).reset_index(drop=True)

    records = []
    n_split_failed = 0
    for i, row in df.iterrows():
        raw_text = clean_text(restore_line_breaks(row[transcription_col]))
        input_text, target_text = split_hospital_course(raw_text)
        if input_text is None:
            n_split_failed += 1
            continue
        records.append({"id": f"mts_{i}", "input": input_text, "target": target_text})

    logger.info(
        f"MTSamples section split: {len(records)} succeeded, "
        f"{n_split_failed} dropped (no hospital-course-like section found)"
    )

    # MTSamples files the same transcription under several specialties (e.g.
    # both "Discharge Summary" and "General Medicine"). Left in, identical
    # notes land in both train and test, inflating test metrics through
    # memorization -- so dedupe on (input, target) before splitting.
    out = pd.DataFrame(records)
    n_before = len(out)
    out = out.drop_duplicates(subset=["input", "target"]).reset_index(drop=True)
    logger.info(f"Removed {n_before - len(out)} duplicate notes (same note filed under multiple specialties)")
    return out


def load_mimic_bhc(cfg: dict) -> pd.DataFrame:
    mb_cfg = cfg["dataset"]["mimic_bhc"]
    raw_path = Path(mb_cfg["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"MIMIC-IV-Ext-BHC file not found at {raw_path}.\n"
            "This dataset requires credentialed PhysioNet access and is never "
            "downloaded automatically -- download it yourself from PhysioNet "
            "and set dataset.mimic_bhc.raw_path in config.yaml."
        )

    logger.info(f"Loading MIMIC-IV-Ext-BHC from {raw_path}")
    df = pd.read_csv(raw_path)

    input_col = _find_column(df, [mb_cfg["input_column"], "input", "text", "note_text"])
    target_col = _find_column(df, [mb_cfg["target_column"], "target", "brief_hospital_course", "bhc"])

    if input_col is None or target_col is None:
        raise ValueError(
            f"Could not find input/target columns in {raw_path}. "
            f"Configured names: input_column={mb_cfg['input_column']!r}, "
            f"target_column={mb_cfg['target_column']!r}. "
            f"Available columns: {list(df.columns)}. "
            "Update these in config.yaml to match your export."
        )

    df = df.dropna(subset=[input_col, target_col]).reset_index(drop=True)

    id_col = mb_cfg.get("id_column")
    ids = df[id_col].astype(str) if id_col and id_col in df.columns else df.index.astype(str)

    records = {
        "id": ids,
        "input": df[input_col].map(clean_text),
        "target": df[target_col].map(clean_text),
    }
    return pd.DataFrame(records).reset_index(drop=True)


# ============================================================================
# Token-length analysis, truncation accounting, split, stats
# ============================================================================

def add_token_lengths(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """
    Tokenize with BioGPT's own tokenizer (not a word count) so length stats
    and truncation decisions downstream are measured in the units that
    actually matter: tokens the model consumes, out of its 1024-token
    context window.
    """
    df = df.copy()
    df["input_tokens"] = df["input"].map(lambda t: len(tokenizer.encode(t)))
    df["target_tokens"] = df["target"].map(lambda t: len(tokenizer.encode(t)))
    return df


def split_dataset(df: pd.DataFrame, cfg: dict, seed: int) -> dict[str, pd.DataFrame]:
    ds_cfg = cfg["dataset"]
    train_frac, val_frac, test_frac = ds_cfg["train_frac"], ds_cfg["val_frac"], ds_cfg["test_frac"]
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "train/val/test fractions must sum to 1"

    train_df, temp_df = train_test_split(df, train_size=train_frac, random_state=seed, shuffle=True)
    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(temp_df, train_size=relative_val_frac, random_state=seed, shuffle=True)

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def compute_stats(splits: dict[str, pd.DataFrame], cfg: dict) -> dict:
    max_input = cfg["dataset"]["max_input_tokens"]
    max_target = cfg["dataset"]["max_target_tokens"]

    stats = {"dataset_mode": cfg["dataset"]["mode"], "max_input_tokens": max_input, "max_target_tokens": max_target}
    for name, df in splits.items():
        n_input_trunc = int((df["input_tokens"] > max_input).sum())
        n_target_trunc = int((df["target_tokens"] > max_target).sum())
        stats[name] = {
            "n_samples": int(len(df)),
            "avg_input_tokens": float(df["input_tokens"].mean()),
            "avg_target_tokens": float(df["target_tokens"].mean()),
            "max_input_tokens_seen": int(df["input_tokens"].max()),
            "max_target_tokens_seen": int(df["target_tokens"].max()),
            "n_input_truncated": n_input_trunc,
            "pct_input_truncated": round(100 * n_input_trunc / len(df), 2) if len(df) else 0.0,
            "n_target_truncated": n_target_trunc,
            "pct_target_truncated": round(100 * n_target_trunc / len(df), 2) if len(df) else 0.0,
        }
    return stats


def plot_token_histogram(splits: dict[str, pd.DataFrame], out_path: Path) -> None:
    all_df = pd.concat(splits.values(), ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(all_df["input_tokens"], bins=40, color="#4C72B0")
    axes[0].set_title("Input (clinical note) token length")
    axes[0].set_xlabel("tokens")
    axes[0].set_ylabel("count")

    axes[1].hist(all_df["target_tokens"], bins=40, color="#DD8452")
    axes[1].set_title("Target (brief hospital course) token length")
    axes[1].set_xlabel("tokens")

    fig.suptitle("Token length distribution (BioGPT tokenizer)")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    mode = cfg["dataset"]["mode"]
    logger.info(f"Dataset mode: {mode}")
    if mode == "mimic_bhc":
        df = load_mimic_bhc(cfg)
    elif mode == "mtsamples":
        df = load_mtsamples(cfg)
    else:
        raise ValueError(f"Unknown dataset.mode '{mode}'; expected 'mimic_bhc' or 'mtsamples'.")

    logger.info(f"Loaded {len(df)} (input, target) pairs before tokenization/splitting.")

    tokenizer = load_biogpt_tokenizer(cfg["model"]["base_model_name"])
    df = add_token_lengths(df, tokenizer)

    splits = split_dataset(df, cfg, seed=cfg["seed"])
    for name, split_df in splits.items():
        logger.info(f"  {name}: {len(split_df)} examples")

    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    for name, split_df in splits.items():
        out_file = processed_dir / f"{name}.jsonl"
        split_df.to_json(out_file, orient="records", lines=True, force_ascii=False)
        logger.info(f"Saved {out_file}")

    stats = compute_stats(splits, cfg)
    stats_path = Path(cfg["paths"]["outputs_dir"]) / "dataset_stats.json"
    save_json(stats, stats_path)
    logger.info(f"Saved dataset stats to {stats_path}")

    hist_path = Path(cfg["paths"]["figures_dir"]) / "token_length_histogram.png"
    plot_token_histogram(splits, hist_path)
    logger.info(f"Saved token length histogram to {hist_path}")


if __name__ == "__main__":
    main()

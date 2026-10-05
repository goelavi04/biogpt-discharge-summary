"""
Evaluate base vs. fine-tuned BioGPT on the test set generations produced by
generate.py: ROUGE-1/2/L, BERTScore, an approximate hallucination proxy, and
a sampling-reproducibility check.

Inputs:  outputs/test_generations.csv  (from generate.py)
         checkpoints/lora_adapter/     (from train.py, needed for the
                                         reproducibility check, which
                                         generates fresh samples)
Outputs: outputs/metrics.json
         outputs/figures/metrics_comparison.png
         outputs/qualitative_examples.md
"""

import re
from pathlib import Path

# NOT `import evaluate` here: this script is itself named evaluate.py, and
# when it's run directly (`python src/evaluate.py`), Python puts the
# script's own directory FIRST on sys.path -- so `import evaluate` would
# resolve to this very file, not the pip package, and silently fail in
# confusing ways (e.g. AttributeError on .load). We compute ROUGE directly
# with the rouge_score library instead, which sidesteps the collision
# entirely and is the same underlying implementation HF's `evaluate.load("rouge")`
# wraps anyway.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM

from generate import decoding_kwargs, generate_one
from utils import build_prompt_ids, ensure_dir, get_device, get_logger, load_biogpt_tokenizer, load_config, save_json, set_seed

logger = get_logger("evaluate")


# ============================================================================
# Hallucination proxy
#
# IMPORTANT, and repeated in metrics.json and the README: this is a crude
# LEXICAL heuristic, not a validated hallucination detector. It flags
# medical-looking strings in the generated text that don't literally appear
# in the source note. It will have false positives (synonyms, abbreviation
# expansions, correct clinical inferences the model drew from context) and
# false negatives (fabricated claims using vocabulary that happens to also
# appear in the source). Report it as "% of extracted terms not found
# verbatim in source", never as "% hallucinated facts".
# ============================================================================

CAPITALIZED_TERM_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
DRUG_SUFFIX_RE = re.compile(r"\b[a-zA-Z]+(?:mab|pril|olol|statin)\b", re.IGNORECASE)
NUMBER_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|mL|ml|L|kg|mmHg|bpm|%|units?)\b", re.IGNORECASE)

# Capitalized words that are common sentence-starters/stopwords, not medical
# terms -- excluded to reduce (not eliminate) false positives from
# CAPITALIZED_TERM_RE matching the first word of every sentence.
COMMON_WORDS = {
    "The", "Patient", "Pt", "This", "That", "After", "During", "On", "He", "She",
    "They", "His", "Her", "Was", "Were", "With", "For", "And", "But",
}


def extract_medical_terms(text: str) -> set[str]:
    terms = set(m.group(0) for m in CAPITALIZED_TERM_RE.finditer(text)) - COMMON_WORDS
    terms |= set(m.group(0) for m in DRUG_SUFFIX_RE.finditer(text))
    terms |= set(m.group(0) for m in NUMBER_UNIT_RE.finditer(text))
    return terms


def hallucination_pct(generated: str, source: str) -> float | None:
    """% of extracted terms from `generated` that do NOT appear (case-insensitive,
    substring match) anywhere in `source`. Returns None if no terms were extracted
    (nothing to measure) so callers can exclude those rows from the mean rather
    than silently counting them as 0%."""
    terms = extract_medical_terms(generated)
    if not terms:
        return None
    source_lower = source.lower()
    missing = [t for t in terms if t.lower() not in source_lower]
    return 100.0 * len(missing) / len(terms)


def mean_hallucination_pct(generated_col: pd.Series, source_col: pd.Series) -> dict:
    values = [hallucination_pct(g, s) for g, s in zip(generated_col, source_col)]
    scored = [v for v in values if v is not None]
    return {
        "mean_pct_terms_not_in_source": round(float(np.mean(scored)), 2) if scored else None,
        "n_examples_with_extracted_terms": len(scored),
        "n_examples_total": len(values),
    }


# ============================================================================
# ROUGE / BERTScore
# ============================================================================

def get_rouge_scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(preds: list[str], refs: list[str], scorer: rouge_scorer.RougeScorer) -> dict:
    """Mean F-measure per ROUGE variant, averaged over examples."""
    totals = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)  # rouge_score API: score(target, prediction)
        for k in totals:
            totals[k].append(scores[k].fmeasure)
    return {k: round(float(np.mean(v)), 4) for k, v in totals.items()}


def compute_rouge_per_example(preds: list[str], refs: list[str], scorer: rouge_scorer.RougeScorer) -> list[float]:
    return [float(scorer.score(ref, pred)["rougeL"].fmeasure) for pred, ref in zip(preds, refs)]


def truncate_to_bert_window(texts: list[str], model_name: str, max_wordpieces: int = 510) -> list[str]:
    """
    Cut each text at the character offset of its `max_wordpieces`-th wordpiece.

    WHY: Bio_ClinicalBERT's Hub repo has no tokenizer_config.json, so its
    tokenizer's model_max_length is "unlimited" and bert_score (which
    truncates to model_max_length) feeds >512-token texts straight into a
    512-position BERT -- an index error that would silently push every run
    onto the fallback model. 510 leaves room for [CLS]/[SEP]. Cutting by
    character offset (not decode()) keeps the original text byte-for-byte.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    out = []
    for text in texts:
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        out.append(text[: offsets[max_wordpieces - 1][1]] if len(offsets) > max_wordpieces else text)
    return out


def compute_bertscore(preds: list[str], refs: list[str], cfg: dict) -> dict:
    import bert_score

    eval_cfg = cfg["evaluation"]
    preferred = eval_cfg["bertscore_model_preferred"]
    try:
        preds_t = truncate_to_bert_window(preds, preferred)
        refs_t = truncate_to_bert_window(refs, preferred)
        # emilyalsentzer/Bio_ClinicalBERT is a bert-base architecture (12
        # layers) but isn't in bert_score's built-in model->layer lookup
        # table, so num_layers must be supplied explicitly. Layer 9 is
        # bert-score's documented best-correlating layer for bert-base-class
        # models; we reuse it here as a reasonable approximation for this
        # checkpoint rather than a value validated specifically for it.
        P, R, F1 = bert_score.score(preds_t, refs_t, model_type=preferred, num_layers=9, verbose=False)
        model_used = preferred
    except Exception as e:
        logger.warning(f"BERTScore with preferred model '{preferred}' failed ({e}); falling back to "
                        f"'{eval_cfg['bertscore_model_fallback']}'.")
        P, R, F1 = bert_score.score(preds, refs, lang=eval_cfg["bertscore_lang"], verbose=False)
        model_used = eval_cfg["bertscore_model_fallback"]

    return {
        "precision": round(float(P.mean()), 4),
        "recall": round(float(R.mean()), 4),
        "f1": round(float(F1.mean()), 4),
        "model_used": model_used,
    }


# ============================================================================
# Reproducibility check
# ============================================================================

def pairwise_rougeL(a: str, b: str, scorer: rouge_scorer.RougeScorer) -> float:
    """ROUGE-L between two runs, except byte-identical strings count as 1.0.
    rouge_score drops punctuation and scores two empty token lists as 0.0, so
    two identical empty (or punctuation-only) generations -- e.g. the model
    emitting </s> immediately -- would otherwise read as "not reproducible"."""
    if a == b:
        return 1.0
    return compute_rouge_per_example([a], [b], scorer)[0]


def reproducibility_check(model, tokenizer, subsample_df: pd.DataFrame, cfg: dict, device: str, scorer) -> dict:
    """
    For a subsample of test examples: generate greedy twice and sampling 3x
    (fine-tuned model), then measure mean pairwise ROUGE-L between the
    repeated runs. Greedy is expected to come out at (or essentially at) 1.0
    because do_sample=False is deterministic given fixed weights and inputs
    -- but we EMPIRICALLY measure it rather than hardcoding 1.0, per the
    project rule that every number in outputs/ must come from actual
    execution (e.g. this also catches non-determinism from things like
    cuDNN algorithm selection, which do occasionally bite greedy decoding on
    GPU).
    """
    gen_cfg = cfg["generation"]
    max_new_tokens = gen_cfg["max_new_tokens"]
    top_k, top_p = gen_cfg["sampling"]["top_k"], gen_cfg["sampling"]["top_p"]
    seeds = gen_cfg["sampling"]["seeds"]
    extra = decoding_kwargs(cfg)

    greedy_pairwise_scores = []
    sampling_pairwise_scores = []

    for _, row in subsample_df.iterrows():
        prompt_ids = build_prompt_ids(tokenizer, row["input"], cfg)

        greedy_runs = [
            generate_one(model, tokenizer, prompt_ids, device, max_new_tokens, do_sample=False, **extra) for _ in range(2)
        ]
        g_rouge = pairwise_rougeL(greedy_runs[0], greedy_runs[1], scorer)
        greedy_pairwise_scores.append(g_rouge)

        sampling_runs = []
        for seed in seeds:
            set_seed(seed)
            sampling_runs.append(
                generate_one(model, tokenizer, prompt_ids, device, max_new_tokens, do_sample=True, top_k=top_k, top_p=top_p, **extra)
            )
        pair_scores = []
        for i in range(len(sampling_runs)):
            for j in range(i + 1, len(sampling_runs)):
                pair_scores.append(pairwise_rougeL(sampling_runs[i], sampling_runs[j], scorer))
        sampling_pairwise_scores.append(float(np.mean(pair_scores)))

    return {
        "n_examples": len(subsample_df),
        "greedy_mean_pairwise_rougeL": round(float(np.mean(greedy_pairwise_scores)), 4),
        "sampling_mean_pairwise_rougeL": round(float(np.mean(sampling_pairwise_scores)), 4),
        "note": "greedy measured empirically (two runs per example), not assumed to be exactly 1.0",
    }


# ============================================================================
# Figures and qualitative examples
# ============================================================================

def plot_comparison_bar_chart(base_rouge: dict, ft_rouge: dict, base_bertscore: dict, ft_bertscore: dict, out_path: Path) -> None:
    metrics = ["rouge1", "rouge2", "rougeL"]
    base_vals = [base_rouge[m] for m in metrics] + [base_bertscore["f1"]]
    ft_vals = [ft_rouge[m] for m in metrics] + [ft_bertscore["f1"]]
    labels = metrics + ["bertscore_f1"]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, base_vals, width, label="base (zero-shot)", color="#8C8C8C")
    ax.bar(x + width / 2, ft_vals, width, label="fine-tuned (LoRA)", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_title("Base vs. fine-tuned BioGPT: automatic metrics")
    ax.legend()
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_qualitative_examples(df: pd.DataFrame, ft_rougeL_scores: list[float], out_path: Path) -> None:
    df = df.copy()
    df["ft_greedy_rougeL"] = ft_rougeL_scores
    df_sorted = df.sort_values("ft_greedy_rougeL", ascending=False)

    good = df_sorted.head(3)
    bad = df_sorted.tail(3)

    lines = ["# Qualitative Examples", "", "3 highest-ROUGE-L and 3 lowest-ROUGE-L fine-tuned (greedy) generations "
             "on the test set, by `ft_greedy` vs. `reference` ROUGE-L. Hallucination flags use the same approximate "
             "lexical proxy described in outputs/metrics.json -- NOT a validated clinical fact-checker.", ""]

    def render_example(title: str, row: pd.Series, flag_hallucinations: bool) -> list[str]:
        out = [f"## {title} (id={row['id']}, ROUGE-L={row['ft_greedy_rougeL']:.3f})", "",
               "**Reference:**", "```", row["reference"], "```",
               "**Fine-tuned generation (greedy):**", "```", row["ft_greedy"], "```"]
        if flag_hallucinations:
            terms = extract_medical_terms(row["ft_greedy"])
            source_lower = row["input"].lower()
            missing = sorted(t for t in terms if t.lower() not in source_lower)
            if missing:
                out += ["**Terms in generation not found in source note (proxy flag, verify manually):**",
                        "`" + "`, `".join(missing) + "`"]
            else:
                out += ["*No proxy-flagged terms.*"]
        out.append("")
        return out

    lines.append("# Good examples")
    for _, row in good.iterrows():
        lines += render_example("Good", row, flag_hallucinations=False)

    lines.append("# Bad examples")
    for _, row in bad.iterrows():
        lines += render_example("Bad", row, flag_hallucinations=True)

    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# Main
# ============================================================================

def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()

    gens_path = Path(cfg["paths"]["outputs_dir"]) / "test_generations.csv"
    if not gens_path.exists():
        raise FileNotFoundError(f"{gens_path} not found. Run src/generate.py first.")
    df = pd.read_csv(gens_path).fillna("")
    logger.info(f"Loaded {len(df)} test generations.")

    scorer = get_rouge_scorer()

    refs = df["reference"].tolist()
    base_rouge = compute_rouge(df["base_greedy"].tolist(), refs, scorer)
    ft_greedy_rouge = compute_rouge(df["ft_greedy"].tolist(), refs, scorer)
    ft_sampling_rouge = compute_rouge(df["ft_sampling"].tolist(), refs, scorer)
    ft_greedy_rougeL_per_example = compute_rouge_per_example(df["ft_greedy"].tolist(), refs, scorer)
    logger.info(f"ROUGE base_greedy={base_rouge} ft_greedy={ft_greedy_rouge}")

    base_bertscore = compute_bertscore(df["base_greedy"].tolist(), refs, cfg)
    ft_bertscore = compute_bertscore(df["ft_greedy"].tolist(), refs, cfg)
    logger.info(f"BERTScore base={base_bertscore} ft={ft_bertscore}")

    hallucination = {
        "base_greedy": mean_hallucination_pct(df["base_greedy"], df["input"]),
        "ft_greedy": mean_hallucination_pct(df["ft_greedy"], df["input"]),
        "ft_sampling": mean_hallucination_pct(df["ft_sampling"], df["input"]),
    }
    logger.info(f"Hallucination proxy (mean %% terms not in source): {hallucination}")

    # Reproducibility check needs a live model (fresh generations), not just
    # the CSV, so load the fine-tuned model here too.
    logger.info("Loading fine-tuned model for reproducibility check...")
    tokenizer = load_biogpt_tokenizer(cfg["model"]["base_model_name"])
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model_name"])
    adapter_dir = Path(cfg["paths"]["checkpoints_dir"]) / "lora_adapter"
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.to(device)
    model.eval()

    n_repro = min(cfg["evaluation"]["reproducibility_sample_size"], len(df))
    subsample = df.sample(n=n_repro, random_state=cfg["seed"])
    repro_results = reproducibility_check(model, tokenizer, subsample, cfg, device, scorer)
    logger.info(f"Reproducibility: {repro_results}")

    metrics = {
        "dataset_mode": cfg["dataset"]["mode"],
        "n_test_examples": len(df),
        "rouge": {"base_greedy": base_rouge, "ft_greedy": ft_greedy_rouge, "ft_sampling": ft_sampling_rouge},
        "bertscore": {"base_greedy": base_bertscore, "ft_greedy": ft_bertscore},
        "hallucination_proxy": {
            "description": "Approximate lexical proxy: % of extracted medical-looking terms "
                            "(capitalized words, drug-name suffixes, numbers+units) from the generated "
                            "text that do not appear verbatim in the source note. NOT a validated "
                            "hallucination/factuality metric -- see README for limitations.",
            **hallucination,
        },
        "reproducibility": repro_results,
    }
    save_json(metrics, Path(cfg["paths"]["outputs_dir"]) / "metrics.json")
    logger.info("Saved outputs/metrics.json")

    plot_comparison_bar_chart(
        base_rouge, ft_greedy_rouge, base_bertscore, ft_bertscore,
        Path(cfg["paths"]["figures_dir"]) / "metrics_comparison.png",
    )
    logger.info("Saved outputs/figures/metrics_comparison.png")

    write_qualitative_examples(
        df, ft_greedy_rougeL_per_example, Path(cfg["paths"]["outputs_dir"]) / "qualitative_examples.md"
    )
    logger.info("Saved outputs/qualitative_examples.md")


if __name__ == "__main__":
    main()

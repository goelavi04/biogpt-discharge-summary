"""
Zero-shot base BioGPT demo: greedy vs. sampling decoding on three fixed,
synthetic (non-patient) prompts.

WHY this script exists on its own, before any fine-tuning: it's a fast
sanity check that the base model, tokenizer, and environment all work (no
LoRA, no dataset required), AND it produces the evidence for a specific
point worth making in the model card's "Working Principle in Action"
section: greedy decoding is deterministic (rerun it, get the identical
string), sampling is not (rerun it with top_k/top_p, get a different string
each time). For a clinical-text generator, that determinism/variance
distinction is directly relevant to whether outputs are reproducible and
auditable -- not just a decoding-strategy footnote.

Output: outputs/baseline_demo.md
"""

from pathlib import Path

from transformers import pipeline

from utils import ensure_dir, get_device, get_logger, load_config, set_seed

logger = get_logger("baseline_demo")

# Fixed, synthetic prompts -- intentionally generic/templated phrasing with
# no real patient identifiers or real case details, per project rules.
PROMPTS = [
    "Patient presented with shortness of breath and elevated blood pressure. Diagnosis: ",
    "A 65-year-old male admitted with chest pain radiating to the left arm. Hospital course: ",
    "Patient with type 2 diabetes admitted for diabetic ketoacidosis. Treatment included ",
]


def generate_greedy(generator, prompt: str, max_new_tokens: int) -> str:
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=generator.tokenizer.pad_token_id,
    )
    return out[0]["generated_text"]


def generate_sampling(generator, prompt: str, max_new_tokens: int, top_k: int, top_p: float, seed: int) -> str:
    set_seed(seed)  # re-seed immediately before each sampling call so each run is independently reproducible
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1,
        pad_token_id=generator.tokenizer.pad_token_id,
    )
    return out[0]["generated_text"]


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Device: {device}")

    gen_cfg = cfg["generation"]
    max_new_tokens = gen_cfg["max_new_tokens"]
    top_k = gen_cfg["sampling"]["top_k"]
    top_p = gen_cfg["sampling"]["top_p"]
    sampling_seeds = gen_cfg["sampling"]["seeds"]

    logger.info(f"Loading base model {cfg['model']['base_model_name']} ...")
    generator = pipeline(
        "text-generation",
        model=cfg["model"]["base_model_name"],
        device=0 if device == "cuda" else -1,
    )
    if generator.tokenizer.pad_token_id is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token

    lines = [
        "# Baseline Demo: Zero-Shot BioGPT, Greedy vs. Sampling",
        "",
        "Base model: `" + cfg["model"]["base_model_name"] + "` (no fine-tuning). "
        "Prompts are synthetic/templated -- no real patient data.",
        "",
        "**Point being demonstrated:** greedy decoding (`do_sample=False`) is deterministic -- "
        "rerunning it produces the exact same text every time. Sampling "
        f"(`do_sample=True, top_k={top_k}, top_p={top_p}`) is stochastic -- three runs with "
        "three different seeds below produce three different continuations from the identical "
        "prompt. For a system generating clinical text, that variance is a real design "
        "consideration: reproducibility vs. diversity of phrasing.",
        "",
    ]

    for i, prompt in enumerate(PROMPTS, start=1):
        logger.info(f"Prompt {i}/{len(PROMPTS)}: {prompt!r}")
        lines.append(f"## Prompt {i}")
        lines.append(f"> {prompt}")
        lines.append("")

        greedy_text = generate_greedy(generator, prompt, max_new_tokens)
        greedy_text_rerun = generate_greedy(generator, prompt, max_new_tokens)
        greedy_stable = greedy_text == greedy_text_rerun

        lines.append("**Greedy (do_sample=False), run twice to confirm determinism:**")
        lines.append("")
        lines.append("```")
        lines.append(greedy_text)
        lines.append("```")
        lines.append(f"*Identical on rerun: {greedy_stable}*")
        lines.append("")

        lines.append(f"**Sampling (do_sample=True, top_k={top_k}, top_p={top_p}), 3 runs with different seeds:**")
        lines.append("")
        sampling_outputs = []
        for seed in sampling_seeds:
            text = generate_sampling(generator, prompt, max_new_tokens, top_k, top_p, seed)
            sampling_outputs.append(text)
            lines.append(f"*seed={seed}:*")
            lines.append("```")
            lines.append(text)
            lines.append("```")
        all_identical = len(set(sampling_outputs)) == 1
        lines.append(f"*All 3 sampling runs identical: {all_identical} (expected: False)*")
        lines.append("")

    out_path = Path(cfg["paths"]["outputs_dir"]) / "baseline_demo.md"
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""
Generate Brief Hospital Course summaries for the test set with:
  - base_greedy:  zero-shot base BioGPT, greedy decoding
  - ft_greedy:    LoRA fine-tuned BioGPT, greedy decoding
  - ft_sampling:  LoRA fine-tuned BioGPT, top_k/top_p sampling

Output: outputs/test_generations.csv (columns: id, input, reference,
base_greedy, ft_greedy, ft_sampling)

WHY one model object instead of two: loading the full base model twice
(once "pure", once wrapped for fine-tuned generation) would roughly double
GPU memory use, which is tight on a T4's ~15GB. Instead we load the base
model once, wrap it with the trained LoRA adapter via PeftModel, and use
`disable_adapter()` as a context manager to temporarily turn the adapter off
for the base_greedy column. This is numerically identical to using the
untouched base model, at half the memory cost.
"""

from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from utils import build_prompt_ids, get_device, get_logger, load_biogpt_tokenizer, load_config, set_seed

logger = get_logger("generate")


# BioGPT's learned positional embeddings cover 1024 positions; prompt +
# generated tokens must fit inside that or generation fails with an index
# error (a CUDA device-side assert on GPU). A 768-token input plus template
# (~785) + max_new_tokens=256 overshoots, so new tokens are capped per prompt.
MAX_POSITIONS = 1024


def decoding_kwargs(cfg: dict) -> dict:
    """Decoding settings shared by every generation (see config.yaml's generation section)."""
    gen_cfg = cfg["generation"]
    return {
        "min_new_tokens": gen_cfg.get("min_new_tokens", 0),
        "no_repeat_ngram_size": gen_cfg.get("no_repeat_ngram_size", 0),
    }


@torch.no_grad()
def generate_one(model, tokenizer, prompt_ids: list[int], device: str, max_new_tokens: int, do_sample: bool,
                  top_k: int | None = None, top_p: float | None = None,
                  min_new_tokens: int = 0, no_repeat_ngram_size: int = 0) -> str:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    max_new_tokens = min(max_new_tokens, MAX_POSITIONS - input_ids.shape[1])
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min(min_new_tokens, max_new_tokens),
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if do_sample:
        gen_kwargs.update(top_k=top_k, top_p=top_p)

    output_ids = model.generate(**gen_kwargs)
    # Slice off the prompt so we only decode newly generated tokens -- the
    # downstream metrics (ROUGE/BERTScore/hallucination proxy) compare
    # generated SUMMARY text against the reference summary, not against the
    # prompt + summary concatenation.
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Device: {device}")

    tokenizer = load_biogpt_tokenizer(cfg["model"]["base_model_name"])

    test_path = Path(cfg["paths"]["processed_dir"]) / "test.jsonl"
    test_df = pd.read_json(test_path, lines=True)
    max_n = cfg["generation"]["max_test_examples"]
    if max_n is not None:
        test_df = test_df.head(max_n)
    logger.info(f"Generating for {len(test_df)} test examples.")

    logger.info(f"Loading base model {cfg['model']['base_model_name']} ...")
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model_name"])

    adapter_dir = Path(cfg["paths"]["checkpoints_dir"]) / "lora_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"No LoRA adapter found at {adapter_dir}. Run src/train.py first -- "
            "generate.py needs a fine-tuned adapter to produce the ft_greedy/ft_sampling columns."
        )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.to(device)
    model.eval()

    gen_cfg = cfg["generation"]
    max_new_tokens = gen_cfg["max_new_tokens"]
    top_k, top_p = gen_cfg["sampling"]["top_k"], gen_cfg["sampling"]["top_p"]
    sampling_seed = gen_cfg["sampling"]["seed"]
    extra = decoding_kwargs(cfg)

    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
        prompt_ids = build_prompt_ids(tokenizer, row["input"], cfg)

        with model.disable_adapter():
            base_greedy = generate_one(model, tokenizer, prompt_ids, device, max_new_tokens, do_sample=False, **extra)

        ft_greedy = generate_one(model, tokenizer, prompt_ids, device, max_new_tokens, do_sample=False, **extra)

        set_seed(sampling_seed)
        ft_sampling = generate_one(
            model, tokenizer, prompt_ids, device, max_new_tokens, do_sample=True, top_k=top_k, top_p=top_p, **extra
        )

        rows.append(
            {
                "id": row["id"],
                "input": row["input"],
                "reference": row["target"],
                "base_greedy": base_greedy,
                "ft_greedy": ft_greedy,
                "ft_sampling": ft_sampling,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(cfg["paths"]["outputs_dir"]) / "test_generations.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()

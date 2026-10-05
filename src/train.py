"""
LoRA fine-tune BioGPT on (clinical note -> brief hospital course) pairs
using the HF Trainer.

Output:
  checkpoints/lora_adapter/        -- the trained LoRA adapter (small, a few MB)
  outputs/figures/loss_curve.png
  outputs/trainer_state.json
  outputs/train_summary.json       -- param counts, best eval loss, timing (model card evidence)
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, EarlyStoppingCallback, Trainer, TrainingArguments

from utils import (
    build_prompt_ids,
    ensure_dir,
    get_device,
    get_logger,
    load_biogpt_tokenizer,
    load_config,
    save_json,
    set_seed,
)

logger = get_logger("train")


def print_named_linear_modules(model) -> None:
    """
    Print every nn.Linear submodule name in the model before we configure
    LoRA. This is a deliberate runtime sanity check, not decoration: the
    LoRA target_modules list in config.yaml ("q_proj", "k_proj", "v_proj",
    "out_proj") was determined by reading BioGPT's source
    (BioGptAttention), but source code and the installed package version can
    drift. Printing the real module names here means a mismatch (e.g. a
    future transformers version renaming these) fails loudly and early
    instead of silently training zero LoRA parameters.
    """
    linear_names = sorted(
        {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    )
    logger.info(f"nn.Linear submodule names found in model: {linear_names}")


def build_tokenize_fn(tokenizer, cfg):
    max_target_tokens = cfg["dataset"]["max_target_tokens"]
    eos_id = tokenizer.eos_token_id

    # Total sequence cap: BioGPT's context is 1024 tokens. max_input_tokens
    # (768, enforced inside build_prompt_ids) + max_target_tokens (256) +
    # a handful of literal template tokens ("### Clinical Note:\n" etc.)
    # routinely adds up to just over 1024 -- so this second, hard cap below
    # is not a rare edge case, it fires on most examples and trims a small
    # amount (~10-15 tokens) off the END of the prompt (closest to the
    # template boundary, furthest from the target) rather than off the
    # target itself, which must stay intact since it's what we're training
    # the model to produce.
    hard_cap = 1024

    def tokenize_fn(example):
        prompt_ids = build_prompt_ids(tokenizer, example["input"], cfg)
        target_ids = tokenizer.encode(example["target"], add_special_tokens=False)[:max_target_tokens]
        target_ids = target_ids + [eos_id]

        overflow = len(prompt_ids) + len(target_ids) - hard_cap
        if overflow > 0:
            prompt_ids = prompt_ids[overflow:]

        input_ids = prompt_ids + target_ids
        # WHY mask the prompt: labels=-100 on every prompt token means the
        # cross-entropy loss is computed ONLY on the target (BHC) tokens.
        # Without this, the model would also be trained to predict the
        # clinical note itself (next-token prediction over the input), which
        # (a) wastes capacity on a task we don't care about -- reproducing
        # the input verbatim -- and (b) makes the loss value depend on input
        # length, so it's not comparable across examples or useful as an
        # early-stopping signal for summarization quality specifically.
        labels = [-100] * len(prompt_ids) + target_ids

        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    return tokenize_fn


class PadCollator:
    """
    Pads input_ids/attention_mask/labels to the longest sequence in each
    batch. We write this by hand instead of using DataCollatorForLanguageModeling
    because that collator assumes labels == input_ids (standard CLM) and has
    no notion of our prompt-masked labels; it would overwrite the -100s we
    just carefully computed above.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)  # -100 on padding too: never compute loss on pad
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
    }


def plot_loss_curve(log_history: list[dict], out_path: Path) -> None:
    train_points = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_points = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if train_points:
        steps, losses = zip(*train_points)
        ax.plot(steps, losses, label="train loss (per logging step)", color="#4C72B0")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")

    if eval_points:
        ax2 = ax.twiny()
        epochs, eval_losses = zip(*eval_points)
        ax2.plot(epochs, eval_losses, "o-", label="val loss (per epoch)", color="#DD8452")
        ax2.set_xlabel("epoch")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels() if eval_points else ([], []))
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.set_title("Training / validation loss")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Device: {device}")
    if device == "cpu":
        logger.warning("No GPU detected. LoRA fine-tuning on CPU will be extremely slow; this is only "
                        "sane for a tiny smoke-test run (e.g. a handful of examples).")

    tokenizer = load_biogpt_tokenizer(cfg["model"]["base_model_name"])

    processed_dir = Path(cfg["paths"]["processed_dir"])
    data_files = {"train": str(processed_dir / "train.jsonl"), "validation": str(processed_dir / "val.jsonl")}
    raw_datasets = load_dataset("json", data_files=data_files)

    tokenize_fn = build_tokenize_fn(tokenizer, cfg)
    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing + masking prompt tokens",
    )

    logger.info(f"Loading base model {cfg['model']['base_model_name']} ...")
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model_name"])
    print_named_linear_modules(base_model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    param_counts = count_parameters(model)
    logger.info(f"Parameter counts: {param_counts}")

    train_cfg = cfg["training"]
    if train_cfg["gradient_checkpointing"]:
        # Required when combining gradient checkpointing with a mostly-frozen
        # (PEFT) base model: checkpointing recomputes activations during the
        # backward pass, but it needs SOME tensor in the graph with
        # requires_grad=True to anchor gradient flow. With the base model
        # frozen, the input embeddings are the only candidate -- this call
        # explicitly turns on requires_grad for them. Omitting it is a
        # well-known silent failure mode (loss.backward() raises or trains
        # nothing) when using peft + gradient_checkpointing together.
        model.enable_input_require_grads()

    checkpoints_dir = ensure_dir(cfg["paths"]["checkpoints_dir"])

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir / "run"),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        fp16=train_cfg["fp16"] and device == "cuda",
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        logging_steps=train_cfg["logging_steps"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        save_total_limit=train_cfg["save_total_limit"],
        report_to=[],  # no wandb/etc by default; keep this self-contained for Colab
        seed=cfg["seed"],
    )

    # Early stopping: num_train_epochs is an upper bound; stop once val loss
    # hasn't improved for `early_stopping_patience` epochs. Combined with
    # load_best_model_at_end, the saved adapter is the best epoch, not the last.
    callbacks = []
    if train_cfg.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"]))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=PadCollator(pad_token_id=tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed_minutes = (time.time() - start_time) / 60

    adapter_dir = checkpoints_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"Saved best LoRA adapter to {adapter_dir}")

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    save_json(trainer.state.log_history, outputs_dir / "trainer_state.json")
    plot_loss_curve(trainer.state.log_history, Path(cfg["paths"]["figures_dir"]) / "loss_curve.png")

    eval_losses = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    train_summary = {
        "base_model": cfg["model"]["base_model_name"],
        "lora_config": lora_cfg,
        "training_config": {k: v for k, v in train_cfg.items()},
        "param_counts": param_counts,
        "final_train_loss": train_result.training_loss,
        "best_eval_loss": min(eval_losses) if eval_losses else None,
        "eval_loss_by_epoch": eval_losses,
        "training_minutes": round(elapsed_minutes, 2),
        "device": device,
    }
    save_json(train_summary, outputs_dir / "train_summary.json")
    logger.info(f"Saved {outputs_dir / 'train_summary.json'}")
    logger.info(f"Training complete in {elapsed_minutes:.1f} min. Best eval loss: {train_summary['best_eval_loss']}")


if __name__ == "__main__":
    main()

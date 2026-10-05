# BioGPT Discharge Summary Drafter

Fine-tunes [`microsoft/biogpt`](https://huggingface.co/microsoft/biogpt) with **LoRA** to write the
**Brief Hospital Course** section of a discharge summary from the rest of the clinical note. The
project includes a reproducible training and evaluation pipeline (Colab T4), a
[model card](MODEL_CARD.md), and a **local web app** that drafts a hospital course from a pasted
note.

![App screenshot](docs/app_screenshot.png)

> **Research prototype. Not for clinical use.** The model was trained on ~100 public sample
> notes. Its drafts are fluent but often contain wrong or invented details (see the screenshot:
> "right *upper* lobe" when the note says *lower*). Every output must be checked against the
> source note.

## Results

Test set of 11 held-out notes ([`outputs/metrics.json`](outputs/metrics.json)):

| Metric | Base BioGPT (zero-shot) | **Fine-tuned (LoRA)** |
|---|---|---|
| ROUGE-1 | 0.038 | **0.273** |
| ROUGE-2 | 0.007 | **0.078** |
| ROUGE-L | 0.026 | **0.175** |
| BERTScore-F1 (Bio_ClinicalBERT) | 0.526 | **0.663** |
| Terms not found in source note (lexical proxy; lower is better) | 64.8% | **40.1%** |

- **Trainable parameters:** 3.1M of 350M (0.9%).
- **Training:** 3.6 minutes on a T4, early-stopped at epoch 13 (best checkpoint: epoch 10,
  validation loss 2.921).
- **Reproducibility:** greedy decoding gave identical output on every rerun (pairwise ROUGE-L
  1.000), while sampling gave 0.172.

The full story is in the **[model card](MODEL_CARD.md)**: how it works, data, training,
evaluation, failure modes, and limitations.

## Run the app

Requirements: Windows with Python 3.10-3.12 and [PyTorch](https://pytorch.org/get-started/locally/)
installed. CPU is fine. The trained adapter is included in `checkpoints/lora_adapter/`, so no
training is needed.

```bash
git clone https://github.com/goelavi04/biogpt-discharge-summary.git
cd biogpt-discharge-summary
start_app.bat
```

`start_app.bat` creates a virtual environment and installs `requirements.txt` on first run. It
then starts the server and opens http://127.0.0.1:8000 once the model is loaded. The first start
also downloads the BioGPT base weights (~1.6 GB). On other platforms, run:

```bash
pip install -r requirements.txt
python app/server.py        # then open http://127.0.0.1:8000
```

Using the app:
1. Paste a note, or click **Load sample note** for a fictional one.
2. Click **Generate draft**. It takes about 20-25 seconds on a laptop CPU.
3. Check the highlighted terms: they appear in the draft but not in your note, so verify them.
   This is a text-matching check and misses wrong-but-real words. **Deterministic** mode gives
   the same draft for the same note every time; **Varied** rewords on each run.

Privacy: the server listens on `127.0.0.1` only (not reachable from other machines). The page
loads nothing from the internet, and note and summary text is never logged; only lengths and
timings are.

## Project structure

```
biogpt-discharge-summary/
├── MODEL_CARD.md               # model card: data, training, evaluation, limitations
├── config.yaml                 # every hyperparameter, path and dataset choice (single source of truth)
├── requirements.txt
├── start_app.bat               # one-click launcher for the app (Windows)
├── app/
│   ├── server.py               # FastAPI backend: note in -> draft hospital course out
│   └── static/index.html       # single-page frontend (self-contained, no external requests)
├── src/
│   ├── utils.py                # seeding, config loading, tokenizer, prompt building, I/O
│   ├── data_prep.py            # load, clean, dedupe, split, build (input -> target) pairs
│   ├── baseline_demo.py        # zero-shot base BioGPT: greedy vs. sampling
│   ├── train.py                # LoRA fine-tuning (HF Trainer, early stopping)
│   ├── generate.py             # test-set generation: base vs. fine-tuned, greedy vs. sampling
│   └── evaluate.py             # ROUGE, BERTScore, hallucination proxy, reproducibility
├── notebooks/
│   ├── colab_runner.ipynb            # clean notebook: runs the whole pipeline on Colab
│   └── colab_runner_executed.ipynb   # the actual run behind the results, with all cell outputs
├── checkpoints/lora_adapter/   # the trained LoRA adapter (12.6 MB)
├── outputs/                    # every metric, figure and generation from that run
└── docs/app_screenshot.png
```

## Reproduce the training

**Colab (recommended):**
1. Open `notebooks/colab_runner.ipynb` in [Colab](https://colab.research.google.com) via
   File → Upload notebook.
2. Set Runtime → Change runtime type → **T4 GPU**.
3. Run all cells. Point `REPO_URL` in the second code cell at this repo to clone it, or upload a
   zip of the project.

The notebook mounts Google Drive so checkpoints survive a dropped session. Its last cell
downloads `biogpt_results.zip`; unzip it into the project folder to use the new model in the
app. The executed version of the run behind the results above is in
`notebooks/colab_runner_executed.ipynb`.

**Locally**, step by step (training on CPU is only practical as a smoke test):

```bash
python src/data_prep.py      # -> data/processed/*.jsonl, outputs/dataset_stats.json, figures/token_length_histogram.png
python src/baseline_demo.py  # -> outputs/baseline_demo.md  (no training required)
python src/train.py          # -> checkpoints/lora_adapter/, outputs/train_summary.json, trainer_state.json, figures/loss_curve.png
python src/generate.py       # -> outputs/test_generations.csv
python src/evaluate.py       # -> outputs/metrics.json, figures/metrics_comparison.png, qualitative_examples.md
```

Every script resolves `config.yaml` and `outputs/` relative to the repo root, so they work from
any working directory. Set `BIOGPT_CONFIG=/path/to/other.yaml` to point every script at a
different config (for example a smoke test with a tiny model and its own `paths:`).
`requirements.txt` deliberately does not pin `torch`: Colab ships a matched CUDA build, and
pinning a different one is a common way to break the GPU runtime.

## Dataset

Set `dataset.mode` in `config.yaml` to one of:

- **`mimic_bhc`** -- [MIMIC-IV-Ext-BHC](https://physionet.org/) from PhysioNet. This requires
  credentialed access that **you** must obtain and download yourself; nothing in this repo
  downloads it automatically, and the data is never committed to git (`data/` is gitignored) --
  redistributing it would violate PhysioNet's Data Use Agreement. Point
  `dataset.mimic_bhc.raw_path` at your local export, and check
  `dataset.mimic_bhc.input_column` / `target_column` match your export's actual column names.
- **`mtsamples`** -- the public fallback. Loads the Kaggle "Medical Transcriptions" CSV (or a
  Hugging Face mirror if no local CSV is found), optionally filters by specialty (`keep_specialties`; default: all),
  and splits each note into (rest-of-note, hospital-course-section) via regex on common section
  headers (`HOSPITAL COURSE:`, `BRIEF HOSPITAL COURSE:`, etc.). This is a **heuristic**: rows
  where no matching header is found are dropped and counted, not guessed at. Check
  `outputs/dataset_stats.json` after running `data_prep.py` to see the drop rate.
  Two quirks of the public export are handled explicitly: it flattens every line break into a
  comma (`...(TAH).,COMPLICATIONS:, None.,HOSPITAL COURSE: , The patient...`), so `data_prep.py`
  restores line breaks before looking for headers; and it files the same note under several
  specialties, so exact duplicates are removed **before** the split (otherwise identical notes
  leak into both train and test). Using all specialties (`keep_specialties: null`), this yields
  101 unique pairs (80 train / 10 val / 11 test) -- small, which is exactly why MIMIC-IV-Ext-BHC
  is the intended dataset and MTSamples only the fallback.

Both modes produce the same schema downstream (`id`, `input`, `target`), so every script after
`data_prep.py` is dataset-mode-agnostic.

## Design decisions (the "why", for the model card)

**Loss masking on the prompt.** The training format is:
```
### Clinical Note:
{input}

### Brief Hospital Course:
{target}</s>
```
`train.py` sets `labels = -100` on every prompt token, so cross-entropy is computed only on the
target (BHC) tokens. Without this, the model would also be trained to predict the input note
itself, which wastes capacity on a task nobody wants (reproducing the input) and makes the loss
value depend on input length -- not comparable across examples, and not a clean early-stopping
signal for summarization quality.

**LoRA on attention projections only.** Confirmed against `transformers`' `modeling_biogpt.py`:
`BioGptAttention` defines separate `q_proj` / `k_proj` / `v_proj` / `out_proj` linear layers (not
GPT-2-style fused `c_attn`). `train.py` targets exactly these four (r=16, alpha=32, dropout=0.05),
leaving the `fc1`/`fc2` feed-forward layers frozen -- this keeps the adapter small (a few million
params vs. BioGPT's 347M) and focused on how the model attends to the input, which is the standard
first thing to try before touching the FFN layers. `train.py` also prints every `nn.Linear`
submodule name at runtime as a sanity check, in case a future `transformers` version renames
something.

**Greedy vs. sampling, everywhere.** `baseline_demo.py` runs both decoding strategies on the base
model before any fine-tuning, to make a concrete point relevant to clinical text: greedy
(`do_sample=False`) is deterministic -- the identical prompt produces the identical output every
time -- while sampling (`top_k=50, top_p=0.95`) varies between runs. `evaluate.py`'s
reproducibility check puts a number on this (mean pairwise ROUGE-L between repeated runs), measured
empirically rather than assumed, including for greedy -- which is expected to land at ~1.0 but is
still actually run twice and scored, not hardcoded.

**Hallucination proxy is a proxy, not a metric.** `evaluate.py` extracts medical-looking terms
from generated text (capitalized words, drug-name suffixes like `-mab`/`-pril`/`-olol`/`-statin`,
numbers with units) and reports the percentage not found verbatim in the source note. This has
real false positives (synonyms, correct clinical inferences, abbreviation expansions) and false
negatives (fabrications using in-source vocabulary). Every place this number appears -- code
comments, `metrics.json`, `qualitative_examples.md` -- labels it as an approximate lexical proxy.
Do not present it in the model card as a validated hallucination rate.

**Why `evaluate.py` uses `rouge_score` directly instead of HF's `evaluate.load("rouge")`.**
This script is itself named `evaluate.py`. When you run `python src/evaluate.py`, Python puts the
script's own directory first on `sys.path`, so `import evaluate` would resolve to *this file*, not
the pip package -- a classic same-name-as-a-stdlib/pip-module shadowing bug (try naming a script
`random.py` some day). `evaluate==0.4.3` is still pinned in `requirements.txt` for parity with the
originally-specified stack, but `src/evaluate.py` computes ROUGE with `rouge_score` directly
(the same underlying implementation `evaluate.load("rouge")` wraps) to avoid importing itself.

**Everything is seeded.** `utils.set_seed()` seeds Python's `random`, numpy, and torch (CPU + all
CUDA devices), called at the start of every script. The three sources of randomness in this
project -- the 80/10/10 split, LoRA weight init / dropout, and sampling decoding -- are all made
reproducible by this, with `config.yaml`'s `seed: 42` as the one place to change it.

## Outputs and the model card

| File | Model card section |
|---|---|
| `dataset_stats.json`, `figures/token_length_histogram.png` | Training data |
| `baseline_demo.md` | How it works: greedy vs. sampling in action |
| `train_summary.json`, `trainer_state.json`, `figures/loss_curve.png` | Training procedure |
| `metrics.json`, `figures/metrics_comparison.png` | Evaluation + reproducibility |
| `test_generations.csv`, `qualitative_examples.md` | Qualitative analysis, limitations |

Every number in `outputs/` comes from an actual run of this code. Nothing is written by hand or
estimated.

## Known limitations

- **Tiny dataset.** 80 training notes. Drafts frequently contain fabricated or wrong details,
  and must be checked line by line.
- **Section-splitting heuristic.** MTSamples pairs come from header matching; notes that don't
  follow common header conventions are dropped (the drop rate is reported).
- **Lexical checks only.** The hallucination proxy and the app's highlights are lexical, not
  semantic or clinically validated.
- **Context limit.** BioGPT's 1024-token context truncates long notes and targets (caps: 768
  input / 256 target tokens).
- **Not a clinical tool.** This is a research and teaching artifact. It is not clinically
  validated and should not inform patient care.

## Acknowledgements

Built on [BioGPT](https://github.com/microsoft/BioGPT) (Microsoft Research),
[PEFT](https://github.com/huggingface/peft), [MTSamples](https://mtsamples.com), and
[Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) for BERTScore.

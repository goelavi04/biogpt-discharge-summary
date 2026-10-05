"""
Local web app: paste a clinical note, get a draft Brief Hospital Course from
the LoRA fine-tuned BioGPT.

    python app/server.py          (or double-click start_app.bat)
    -> http://127.0.0.1:8000

Needs checkpoints/lora_adapter/ (from train.py / the Colab results zip) and
the microsoft/biogpt base weights (downloaded to the Hugging Face cache on
first start).

WHY the server binds to 127.0.0.1 only and never logs note text: notes typed
into this app may be real clinical text. Binding to localhost means nothing
else on the network can reach it, and keeping note contents out of the logs
means nothing is written to disk -- the text lives only in this process's
memory for the duration of one request.
"""

# torch MUST be imported before pandas (pulled in via generate.py /
# evaluate.py below): on some Windows setups importing pandas first makes
# torch's c10.dll fail to initialize (WinError 1114).
import torch  # noqa: F401  (import-order side effect, see above)

import os
import re
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from evaluate import extract_medical_terms  # noqa: E402  (src/evaluate.py, not the pip package)
from generate import decoding_kwargs, generate_one  # noqa: E402
from utils import (  # noqa: E402
    build_prompt_ids,
    get_device,
    get_logger,
    load_biogpt_tokenizer,
    load_config,
    load_json,
    set_seed,
)

logger = get_logger("app")
STATIC_DIR = Path(__file__).resolve().parent / "static"
HOST, PORT = "127.0.0.1", 8000
APP_URL = f"http://{HOST}:{PORT}"

# Long enough for a full discharge summary, short enough to reject accidental
# pastes of whole charts (BioGPT only reads the first max_input_tokens anyway).
MAX_NOTE_CHARS = 20_000


class ModelState:
    """Loaded once at startup; a lock serializes generation because one
    model instance on CPU can't usefully run two generate() calls at once."""

    def __init__(self):
        self.cfg = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.error = None
        self.metrics = None
        self.lock = threading.Lock()


state = ModelState()


def load_model() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    cfg = load_config()
    state.cfg = cfg
    adapter_dir = Path(cfg["paths"]["checkpoints_dir"]) / "lora_adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        state.error = (
            f"No trained adapter found at {adapter_dir}. Unzip biogpt_results.zip (downloaded by the last "
            "cell of notebooks/colab_runner.ipynb) into the project folder so that "
            "checkpoints/lora_adapter/adapter_config.json exists, then restart the app."
        )
        logger.error(state.error)
        return

    state.device = get_device()
    logger.info(f"Loading {cfg['model']['base_model_name']} + adapter on {state.device} "
                "(first start downloads ~1.6 GB of base weights)...")
    state.tokenizer = load_biogpt_tokenizer(cfg["model"]["base_model_name"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model_name"])
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    # Folding the LoRA weights into the base weights makes CPU inference as
    # fast as the plain base model (no extra adapter matmuls per layer).
    state.model = model.merge_and_unload().to(state.device).eval()

    metrics_path = Path(cfg["paths"]["outputs_dir"]) / "metrics.json"
    if metrics_path.exists():
        state.metrics = load_json(metrics_path)
    logger.info("Model ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:  # surface load failures in the UI instead of a dead server
        state.error = f"Model failed to load: {e}"
        logger.exception("Model failed to load")
    # Opening the browser only now (model loaded, server about to accept
    # requests) avoids landing on a "can't reach this page" screen when
    # startup takes longer than a launcher's fixed delay. start_app.bat sets
    # OPEN_BROWSER=1; plain `python app/server.py` doesn't open anything.
    if os.environ.get("OPEN_BROWSER") == "1":
        threading.Timer(1.0, webbrowser.open, args=[APP_URL]).start()
    yield


app = FastAPI(title="BioGPT Discharge Summary", lifespan=lifespan)


class SummarizeRequest(BaseModel):
    note: str = Field(..., min_length=1, max_length=MAX_NOTE_CHARS)
    decoding: str = Field("greedy", pattern="^(greedy|sampling)$")
    seed: int = 0


STANDALONE_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def unverified_terms(summary: str, note: str) -> list[str]:
    """
    Terms in the summary to check by hand: evaluate.py's medical-looking terms
    plus every standalone number, kept only if not found in the note as a
    whole word/number (case-insensitive).

    Stricter than evaluate.py's proxy on purpose (that one stays unchanged so
    reported metrics remain comparable): its plain substring match passes a
    fabricated "5 mg" because the note contains "875 mg", and it ignores bare
    numbers, so an invented "BP 140 / 80" went unflagged. Still lexical: a
    wrong word like "upper" (note: "lower") lobe is NOT caught.
    """
    terms = extract_medical_terms(summary) | set(STANDALONE_NUMBER_RE.findall(summary))

    def in_note(term: str) -> bool:
        return re.search(r"(?<![\w.])" + re.escape(term) + r"(?!\w)", note, re.IGNORECASE) is not None

    return sorted(t for t in terms if not in_note(t))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    info = {"ready": state.model is not None, "error": state.error, "device": state.device}
    if state.cfg:
        info["base_model"] = state.cfg["model"]["base_model_name"]
        info["max_input_tokens"] = state.cfg["dataset"]["max_input_tokens"]
    if state.metrics:
        info["eval"] = {
            "rougeL": state.metrics["rouge"]["ft_greedy"]["rougeL"],
            "bertscore_f1": state.metrics["bertscore"]["ft_greedy"]["f1"],
            "n_test_examples": state.metrics["n_test_examples"],
        }
    return info


@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail=state.error or "Model is still loading.")

    cfg = state.cfg
    gen_cfg = cfg["generation"]
    note = req.note.strip()
    note_tokens = len(state.tokenizer.encode(note, add_special_tokens=False))
    prompt_ids = build_prompt_ids(state.tokenizer, note, cfg)

    start = time.time()
    with state.lock:
        if req.decoding == "sampling":
            set_seed(req.seed)
            summary = generate_one(
                state.model, state.tokenizer, prompt_ids, state.device, gen_cfg["max_new_tokens"],
                do_sample=True, top_k=gen_cfg["sampling"]["top_k"], top_p=gen_cfg["sampling"]["top_p"],
                **decoding_kwargs(cfg),
            )
        else:
            summary = generate_one(
                state.model, state.tokenizer, prompt_ids, state.device, gen_cfg["max_new_tokens"],
                do_sample=False, **decoding_kwargs(cfg),
            )
    seconds = round(time.time() - start, 1)
    # Lengths and timing only -- never the note or summary text (see module docstring).
    logger.info(f"Generated {len(summary)} chars from a {note_tokens}-token note in {seconds}s ({req.decoding})")

    return {
        "summary": summary,
        "unverified_terms": unverified_terms(summary, note),
        "note_tokens": note_tokens,
        "max_input_tokens": cfg["dataset"]["max_input_tokens"],
        "truncated": note_tokens > cfg["dataset"]["max_input_tokens"],
        "decoding": req.decoding,
        "seconds": seconds,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

"""Central configuration: paths, model names and hyper-parameters.

Every script imports from here so there is one place to change settings.
Values can be overridden with environment variables (see .env.example).
"""
import os
from pathlib import Path

try:  # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TAXONOMY_PATH = DATA_DIR / "taxonomy.json"
RAW_DATA_PATH = RAW_DIR / "procurement_raw.csv"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
VAL_PATH = PROCESSED_DIR / "val.jsonl"
TEST_PATH = PROCESSED_DIR / "test.jsonl"
CATEGORIZED_PATH = OUTPUT_DIR / "categorized_spend.csv"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", MODELS_DIR / "llama3-spend-qlora"))

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- base LLM
# "unsloth/llama-3-8b-Instruct" is an ungated copy of Meta's
# Llama-3-8B-Instruct weights, so no Hugging Face token is needed.
# If you have accepted Meta's licence on Hugging Face you can instead use
# BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct and set HF_TOKEN.
BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/llama-3-8b-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or None

# ---------------------------------------------------------------- QLoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
MAX_SEQ_LEN = 320
SEED = 42

# ---------------------------------------------------------------- inference
REVIEW_CONFIDENCE_THRESHOLD = 0.60  # predictions below this get flagged

# ---------------------------------------------------------------- agent LLM
# rules  -> no API key, built-in question parser (always works)
# gemini -> Google AI Studio key (free tier)  GEMINI_API_KEY
# groq   -> Groq key (free tier)              GROQ_API_KEY
# ollama -> local Ollama server, no key        (ollama pull llama3.1:8b)
AGENT_BACKEND = os.getenv("AGENT_BACKEND", "rules")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

from pathlib import Path
from masfactory import OpenAIModel, LegacyOpenAIModel
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Vision Agent settings
MODEL_PATH = PROJECT_ROOT / "models" / "modello_cnn_vine_disease-AdaptiveAvgPool2d--99.5.ckpt"
DATASET_DIR = PROJECT_ROOT / "dataset" / "Dataset-splittato" / "train"
IMAGE_SIZE = 256
VISION_TOP_K = 4

DEFAULT_LLM_MODEL = LegacyOpenAIModel(
    model_name="google/gemma-4-31b-it",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# Context Agent settings
DEFAULT_LOCATION = "Piemonte"
DEFAULT_GROWTH_STAGE = "sconosciuto"
DEFAULT_WEATHER_RISK = "medio"


# Embedding settings
GUIDELINE_CHUNKS_PATH = PROJECT_ROOT / "data" / "guidelines" / "chunks" / "guideline_chunks.jsonl"
PRODUCT_CHUNKS_PATH = PROJECT_ROOT / "data" / "products" / "chunks" / "product_chunks.jsonl"

QDRANT_LOCAL_PATH = PROJECT_ROOT / "vector_store" / "qdrant_local"

QDRANT_COLLECTION_GUIDELINES = "vine_guidelines"
QDRANT_COLLECTION_PRODUCTS = "plant_protection_products"

EMBEDDING_BACKEND = "openvino"

EMBEDDING_MODEL_HF_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "embeddings" / "qwen3-embedding-0.6b-openvino"

# Usa prima il modello locale OpenVINO (pre-esportato).
EMBEDDING_MODEL_NAME = str(EMBEDDING_MODEL_LOCAL_PATH)

# Se True, in assenza del modello locale usa Hugging Face (con possibile export on-the-fly).
# Se False, fallisce con errore esplicito e invita a lanciare lo script di export.
EMBEDDING_ALLOW_HF_FALLBACK = False
EMBEDDING_BATCH_SIZE = 4
EMBEDDING_VECTOR_SIZE = 1024

# Variante più pesante, da provare dopo
# EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
# EMBEDDING_VECTOR_SIZE = 2560
# EMBEDDING_BATCH_SIZE = 1

OPENVINO_DEVICE = "GPU"

# RAG settings
RAG_TOP_K_GUIDELINES = 8
RAG_TOP_K_PRODUCTS = 5

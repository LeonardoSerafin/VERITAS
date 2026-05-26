from pathlib import Path
from masfactory import OpenAIModel, LegacyOpenAIModel
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Vision Agent settings

MODEL_PATH = PROJECT_ROOT / "models" / "modello_cnn_vine_disease-AdaptiveAvgPool2d--99.5.ckpt"
DATASET_DIR = PROJECT_ROOT / "example_dataset"
CLASS_NAMES = [
    "Black Rot (Guignardia bidwelii)", 
    "ESCA", 
    "Sana", 
    "Escoriosi (Phomopsis viticola)"
    ]
IMAGE_SIZE = 256
VISION_TOP_K = 4

DEFAULT_LLM_MODEL = LegacyOpenAIModel(
    model_name="google/gemini-3.1-flash-lite",
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

# Runtime embedding:
# - "auto": sceglie OpenVINO locale su GPU Intel, altrimenti PyTorch.
# - "openvino": forza OpenVINO locale.
# - "torch": forza PyTorch con device automatico.
#
# Questi valori sono configurazioni interne, non backend SentenceTransformers.
EMBEDDING_RUNTIME = "auto"
EMBEDDING_TORCH_DEVICE = "auto"
OPENVINO_DEVICE = "GPU"

# Usato solo quando EMBEDDING_RUNTIME="openvino": se False, l'assenza
# del modello locale OpenVINO produce errore esplicito invece del fallback PyTorch.
EMBEDDING_ALLOW_HF_FALLBACK = False

# Embedding model 

# EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# EMBEDDING_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "embeddings" / "qwen3-embedding-0.6b-openvino"
# EMBEDDING_BATCH_SIZE = 4
# EMBEDDING_VECTOR_SIZE = 1024

# Variante più pesante, da provare dopo
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "embeddings" / "qwen3-embedding-4b-openvino"
EMBEDDING_VECTOR_SIZE = 2560
EMBEDDING_BATCH_SIZE = 4



# RAG settings

RAG_RERANK_ENABLED = True
# Riduce il picco RAM/VRAM: embedder e reranker vengono scaricati dopo l'uso.
RAG_RELEASE_MODELS_AFTER_USE = True

RAG_CANDIDATE_TOP_K_GUIDELINES = 50
RAG_CANDIDATE_TOP_K_PRODUCTS = 50

# Reranker model
# RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
# RERANKER_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "rerankers" / "qwen3-reranker-0.6b-openvino"
# RERANKER_BATCH_SIZE = 4

RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
RERANKER_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "rerankers" / "qwen3-reranker-4b-openvino"
RERANKER_BATCH_SIZE = 4

# Runtime reranker:
# - "auto": sceglie OpenVINO locale su GPU Intel, altrimenti PyTorch.
# - "openvino": forza OpenVINO locale.
# - "torch": forza PyTorch con device automatico.
RERANKER_RUNTIME = "auto"
RERANKER_TORCH_DEVICE = "auto"
RERANKER_OPENVINO_DEVICE = "GPU"
RERANKER_ALLOW_HF_FALLBACK = False

RERANKER_INSTRUCTION = (
    "Given a query about viticulture, grapevine diseases, agronomic guidelines, "
    "plant protection products, active substances, legal constraints, treatment timing, "
    "and safety rules, judge whether the document passage is relevant."
)

RAG_TOP_K_GUIDELINES = 8
RAG_TOP_K_PRODUCTS = 5

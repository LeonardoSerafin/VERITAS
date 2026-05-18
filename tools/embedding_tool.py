from __future__ import annotations
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from config import settings


class EmbeddingTool:
    def __init__(
        self,
        model_name: str,
        device: str = settings.OPENVINO_DEVICE,
        fallback_model_name: str | None = settings.EMBEDDING_MODEL_HF_NAME,
        allow_fallback: bool = settings.EMBEDDING_ALLOW_HF_FALLBACK,
    ):
        """
        Tool unico per Qwen3 Embedding via OpenVINO.

        model_name esempi:
        - Qwen/Qwen3-Embedding-0.6B
        - Qwen/Qwen3-Embedding-4B

        device esempi:
        - GPU
        - CPU
        - AUTO
        """
        self.model_name = model_name
        self.device = device
        self.fallback_model_name = fallback_model_name
        self.allow_fallback = allow_fallback

        resolved_model_name, local_files_only = self._resolve_model_name()
        is_local_model = local_files_only

        model_kwargs = {
            "device": device,
        }
        processor_kwargs = {
            # Evita warning/edge case noti con tokenizer Mistral regex validation in transformers recenti.
            "fix_mistral_regex": True,
        }

        if is_local_model:
            # Forza il caricamento dell'IR locale evitando inferenze automatiche su export/subfolder.
            local_root = Path(resolved_model_name)
            cache_dir = local_root / "openvino" / "model_cache"
            model_kwargs.update(
                {
                    "export": False,
                    "subfolder": "openvino",
                    "file_name": "openvino_model.xml",
                    "ov_config": {
                        "CACHE_DIR": str(cache_dir),
                    },
                }
            )

        self.model = SentenceTransformer(
            resolved_model_name,
            backend="openvino",
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )

    @staticmethod
    def _looks_like_hf_repo_id(name: str) -> bool:
        # Esempio: "Qwen/Qwen3-Embedding-0.6B"
        return "/" in name and "\\" not in name and not Path(name).is_absolute()

    def _resolve_model_name(self) -> tuple[str, bool]:
        local_path = Path(self.model_name)
        if local_path.exists():
            return str(local_path), True

        if self._looks_like_hf_repo_id(self.model_name):
            return self.model_name, False

        if self.allow_fallback and self.fallback_model_name:
            return self.fallback_model_name, False

        raise FileNotFoundError(
            "Modello embedding OpenVINO locale non trovato.\n"
            f"Path atteso: {local_path}\n"
            "Esegui prima: python scripts/export_embedding_openvino.py\n"
            "Oppure abilita EMBEDDING_ALLOW_HF_FALLBACK=True in config/settings.py."
        )

    def _format_query(self, query: str) -> str:
        """
        Qwen3 Embedding supporta instruction-aware retrieval.

        Usiamo una instruction fissa per il tuo caso:
        recuperare passaggi rilevanti da documenti agronomici,
        disciplinari e registri di prodotti fitosanitari.
        """
        return (
            "Instruct: Given a query about viticulture, grapevine diseases, "
            "agronomic guidelines, plant protection products, active substances, "
            "legal constraints, treatment timing, and safety rules, retrieve the "
            "most relevant passages from technical documents and product registry records.\n"
            f"Query: {query}"
        )

    def embed_texts(self, texts: List[str], batch_size: int = settings.EMBEDDING_BATCH_SIZE) -> List[List[float]]:
        """
        Embedding dei documenti/chunk.

        Qui NON aggiungiamo instruction ai documenti.
        I chunk devono restare nel loro contenuto naturale.
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embedding della query.

        Qui invece aggiungiamo l'instruction, perché Qwen3
        può sfruttarla per retrieval task-specific.
        """
        formatted_query = self._format_query(query)

        vector = self.model.encode(
            [formatted_query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return vector.tolist()

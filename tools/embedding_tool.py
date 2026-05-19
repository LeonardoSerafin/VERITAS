from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from config import settings


@dataclass(frozen=True)
class EmbeddingRuntimeInfo:
    runtime: str
    device: str
    model_name: str
    model_source: str
    details: str = ""


class EmbeddingTool:
    def __init__(
        self,
        model_name: str | None = None,
        torch_device: str | None = None,
        openvino_device: str | None = None,
        runtime: str = settings.EMBEDDING_RUNTIME,
        hf_model_name: str = settings.EMBEDDING_MODEL_NAME,
        openvino_model_path: str | Path = settings.EMBEDDING_MODEL_LOCAL_PATH,
        allow_hf_fallback: bool = settings.EMBEDDING_ALLOW_HF_FALLBACK,
    ):
        """
        Tool unico per embedding documenti e query RAG.

        La scelta del runtime è interna:
        - OpenVINO locale viene usato automaticamente se c'e una GPU Intel e
          sono presenti gli artifact esportati.
        - Altrimenti si usa SentenceTransformers/PyTorch con CUDA, MPS o CPU.
        """
        self.requested_model_name = model_name
        self.hf_model_name = hf_model_name
        self.openvino_model_path = Path(openvino_model_path)
        self.requested_torch_device = (torch_device or settings.EMBEDDING_TORCH_DEVICE or "auto").lower()
        self.requested_openvino_device = (openvino_device or settings.OPENVINO_DEVICE or "GPU").upper()
        self.requested_runtime = (runtime or "auto").lower()
        self.allow_hf_fallback = allow_hf_fallback

        self.runtime_info = self._select_runtime()
        self.model = self._load_model(self.runtime_info)

    @staticmethod
    def _has_openvino_artifacts(path: Path) -> bool:
        return (
            (path / "openvino_model.xml").exists()
            or (path / "openvino" / "openvino_model.xml").exists()
        )

    @staticmethod
    def _load_openvino_core():
        try:
            from openvino._ov_api import Core

            return Core()
        except Exception:
            return None

    @classmethod
    def _find_intel_openvino_gpu(cls) -> tuple[str | None, str]:
        core = cls._load_openvino_core()
        if core is None:
            return None, "OpenVINO non installato o non importabile"

        try:
            devices = list(core.available_devices)
        except Exception as exc:
            return None, f"device OpenVINO non leggibili: {exc}"

        gpu_devices = [
            device
            for device in devices
            if device == "GPU" or device.startswith("GPU.")
        ]

        if not gpu_devices:
            return None, f"nessuna GPU OpenVINO tra i device disponibili: {devices}"

        for device in gpu_devices:
            try:
                full_name = str(core.get_property(device, "FULL_DEVICE_NAME"))
            except Exception:
                full_name = ""

            if "intel" in full_name.lower():
                return device, full_name

        return None, f"GPU OpenVINO presenti ma non Intel: {gpu_devices}"

    @staticmethod
    def _torch_device_available(device: str) -> bool:
        try:
            import torch
        except Exception:
            return device == "cpu"

        if device == "cuda":
            return bool(torch.cuda.is_available())

        if device == "mps":
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            return bool(mps and mps.is_available())

        return device == "cpu"

    def _select_torch_device(self) -> str:
        if self.requested_torch_device in {"cuda", "mps", "cpu"}:
            if self._torch_device_available(self.requested_torch_device):
                return self.requested_torch_device
            return "cpu"

        try:
            import torch
        except Exception:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps and mps.is_available():
            return "mps"

        return "cpu"

    def _torch_model_name(self) -> str:
        if not self.requested_model_name:
            return self.hf_model_name

        requested_path = Path(self.requested_model_name)
        if requested_path == self.openvino_model_path or self._has_openvino_artifacts(requested_path):
            return self.hf_model_name

        return self.requested_model_name

    def _select_runtime(self) -> EmbeddingRuntimeInfo:
        openvino_available = self._has_openvino_artifacts(self.openvino_model_path)

        if self.requested_runtime == "openvino":
            if not openvino_available:
                if not self.allow_hf_fallback:
                    raise FileNotFoundError(
                        "Modello embedding OpenVINO locale non trovato.\n"
                        f"Path atteso: {self.openvino_model_path}\n"
                        "Esegui prima: python scripts/export_embedding_openvino.py --verify\n"
                        "Oppure usa EMBEDDING_RUNTIME=auto/torch."
                    )
                return self._select_torch_runtime(
                    "OpenVINO locale mancante; fallback PyTorch abilitato"
                )

            return EmbeddingRuntimeInfo(
                runtime="openvino",
                device=self.requested_openvino_device,
                model_name=str(self.openvino_model_path),
                model_source="local_openvino",
                details="OpenVINO forzato da configurazione",
            )

        if self.requested_runtime == "torch":
            return self._select_torch_runtime("PyTorch forzato da configurazione")

        if self.requested_runtime != "auto":
            raise ValueError(
                "EMBEDDING_RUNTIME non valido. Usa: auto, openvino, torch."
            )

        intel_gpu, intel_gpu_name = self._find_intel_openvino_gpu()
        if intel_gpu and openvino_available:
            return EmbeddingRuntimeInfo(
                runtime="openvino",
                device=self.requested_openvino_device,
                model_name=str(self.openvino_model_path),
                model_source="local_openvino",
                details=f"GPU Intel rilevata: {intel_gpu_name}",
            )

        fallback_reason = (
            "GPU Intel rilevata ma modello OpenVINO locale mancante"
            if intel_gpu
            else intel_gpu_name
        )
        return self._select_torch_runtime(fallback_reason)

    def _select_torch_runtime(self, details: str = "") -> EmbeddingRuntimeInfo:
        device = self._select_torch_device()
        return EmbeddingRuntimeInfo(
            runtime="torch",
            device=device,
            model_name=self._torch_model_name(),
            model_source="huggingface_or_local",
            details=details,
        )

    def _load_model(self, runtime_info: EmbeddingRuntimeInfo) -> SentenceTransformer:
        processor_kwargs = {
            # Evita warning/edge case noti con tokenizer Mistral regex validation in transformers recenti.
            "fix_mistral_regex": True,
        }

        if runtime_info.runtime == "openvino":
            local_root = Path(runtime_info.model_name)
            cache_dir = local_root / "openvino" / "model_cache"

            return SentenceTransformer(
                str(local_root),
                backend="openvino",
                model_kwargs={
                    "device": runtime_info.device,
                    "export": False,
                    "subfolder": "openvino",
                    "file_name": "openvino_model.xml",
                    "ov_config": {
                        "CACHE_DIR": str(cache_dir),
                    },
                },
                processor_kwargs=processor_kwargs,
                local_files_only=True,
                trust_remote_code=True,
            )

        return SentenceTransformer(
            runtime_info.model_name,
            device=runtime_info.device,
            processor_kwargs=processor_kwargs,
            trust_remote_code=True,
        )

    def describe_runtime(self) -> str:
        info = self.runtime_info
        suffix = f" ({info.details})" if info.details else ""
        return (
            f"runtime={info.runtime}, device={info.device}, "
            f"model_source={info.model_source}, model={info.model_name}{suffix}"
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

        Qui invece aggiungiamo l'instruction, per retrieval task-specific.
        """
        formatted_query = self._format_query(query)

        vector = self.model.encode(
            [formatted_query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return vector.tolist()

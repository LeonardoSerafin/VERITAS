from __future__ import annotations
from masfactory.adapters.retrieval import Retrieval
from masfactory.adapters.context.types import ContextQuery, ContextBlock
import json
from typing import Any, Optional
from qdrant_client.models import FieldCondition, Filter, MatchValue
from config import settings
from tools.embedding_tool import EmbeddingTool
from tools.qdrant_client_factory import get_qdrant_client



class QdrantRetrieval(Retrieval):
    """Adapter for Qdrant vector database retrieval."""

    def __init__(
        self,
        *,
        collection_name: str = settings.QDRANT_COLLECTION_GUIDELINES,
        context_label: str = "GUIDELINES_QDRANT",
        passive: bool = True,
        active: bool = False,
        default_top_k: int = settings.RAG_TOP_K_GUIDELINES,
        min_score: float | None = None,
        default_document_type: str | None = None,
        force_document_type_filter: bool = True,
    ) -> None:
        super().__init__(context_label=context_label, passive=passive, active=active)

        self._collection_name = collection_name
        self._default_top_k = int(default_top_k)
        self._min_score = min_score
        self._default_document_type = default_document_type
        self._force_document_type_filter = force_document_type_filter

        self._client = get_qdrant_client(settings.QDRANT_LOCAL_PATH)
        self._embedding_tool = EmbeddingTool(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.OPENVINO_DEVICE,
        )

    def _query_text_from_context_query(self, query: ContextQuery) -> str:
        text = (query.query_text or "").strip()
        if text:
            return text

        if query.inputs:
            try:
                return json.dumps(query.inputs, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(query.inputs)

        return ""

    def _build_filter(self, query: ContextQuery) -> Optional[Filter]:
        conditions: list[FieldCondition] = []

        # region opzionale da input (es: {"region": "Veneto"})
        region = None
        if query.inputs and isinstance(query.inputs, dict):
            region = query.inputs.get("region")

        if isinstance(region, str) and region.strip():
            conditions.append(
                FieldCondition(
                    key="region",
                    match=MatchValue(value=region.strip()),
                )
            )

        # filtro document_type
        document_type = self._default_document_type
        if query.inputs and isinstance(query.inputs, dict):
            dt = query.inputs.get("document_type")
            if isinstance(dt, str) and dt.strip():
                document_type = dt.strip()

        if self._force_document_type_filter and document_type:
            conditions.append(
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value=document_type),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    def _search_points(
        self,
        *,
        query_vector: list[float],
        query_filter: Optional[Filter],
        limit: int,
    ):
        try:
            return self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except AttributeError:
            return self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            ).points

    def get_blocks(self, query: ContextQuery, *, top_k: int = 8) -> list[ContextBlock]:
        query_text = self._query_text_from_context_query(query)
        if not query_text:
            return []

        k = int(top_k)
        if k <= 0:
            k = self._default_top_k

        query_vector = self._embedding_tool.embed_query(query_text)
        query_filter = self._build_filter(query)

        points = self._search_points(
            query_vector=query_vector,
            query_filter=query_filter,
            limit=k,
        )

        blocks: list[ContextBlock] = []
        for point in points:
            payload: dict[str, Any] = point.payload or {}
            score = float(point.score) if point.score is not None else None

            if self._min_score is not None and score is not None and score < self._min_score:
                continue

            text = payload.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                continue

            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            block = ContextBlock(
                text=text,
                uri=payload.get("source_file"),
                chunk_id=payload.get("chunk_id"),
                score=score,
                title=payload.get("title"),
                metadata=metadata,
                dedupe_key=payload.get("chunk_id") or payload.get("doc_id"),
            )
            blocks.append(block)

        return blocks
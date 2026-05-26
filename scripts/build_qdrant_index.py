from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_LOCAL_PATH,
    EMBEDDING_RUNTIME,
    EMBEDDING_TORCH_DEVICE,
    EMBEDDING_VECTOR_SIZE,
    GUIDELINE_CHUNKS_PATH,
    PRODUCT_CHUNKS_PATH,
    QDRANT_COLLECTION_GUIDELINES,
    QDRANT_COLLECTION_PRODUCTS,
    QDRANT_LOCAL_PATH,
    OPENVINO_DEVICE,
)
from tools.qdrant_client_factory import get_qdrant_client  # noqa: E402
from tools.embedding_tool import EmbeddingTool  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Errore JSON alla riga {line_number} del file {path}"
                ) from e

    return records


def batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def clean_payload(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        return [clean_payload(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): clean_payload(item)
            for key, item in value.items()
            if item is not None
        }

    return str(value)


def build_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata", {})

    payload = {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "source_file": chunk.get("source_file"),
        "title": chunk.get("title"),
        "text": chunk.get("text"),
        "metadata": metadata,
    }

    useful_filter_keys = [
        # Guidelines
        "document_type",
        "region",
        "year",
        "has_table",
        "content_kind",
        "topic",
        "adversities",
        "active_substances",
        "rule_types",
        "legal_relevance",

        # Products
        "status",
        "product_name",
        "registration_number",
        "company",
        "categories",
        "possibly_vine_relevant",
    ]

    for key in useful_filter_keys:
        if key in metadata:
            payload[key] = metadata[key]

    return clean_payload(payload)


def recreate_collection(
    client,
    collection_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    existing_collections = {
        collection.name
        for collection in client.get_collections().collections
    }

    if collection_name in existing_collections:
        if recreate:
            print(f"[INFO] Cancello collection esistente: {collection_name}")
            client.delete_collection(collection_name=collection_name)
        else:
            print(f"[INFO] Collection già esistente: {collection_name}")
            print("[INFO] Verrà aggiornata con upsert.")
            return

    print(f"[INFO] Creo collection: {collection_name}")
    print(f"[INFO] Vector size: {vector_size}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def index_chunks(
    chunks_path: Path,
    collection_name: str,
    recreate: bool,
) -> None:
    print(f"[INFO] Carico chunks da: {chunks_path}")
    chunks = load_jsonl(chunks_path)

    if not chunks:
        raise ValueError(f"Nessun chunk trovato in {chunks_path}")

    print(f"[INFO] Numero chunks: {len(chunks)}")
    print(f"[INFO] Embedding model HF: {EMBEDDING_MODEL_NAME}")
    print(f"[INFO] Embedding runtime richiesto: {EMBEDDING_RUNTIME}")
    print(f"[INFO] PyTorch device richiesto: {EMBEDDING_TORCH_DEVICE}")
    print(f"[INFO] OpenVINO device richiesto: {OPENVINO_DEVICE}")
    print(f"[INFO] OpenVINO locale: {EMBEDDING_MODEL_LOCAL_PATH}")

    client = get_qdrant_client(QDRANT_LOCAL_PATH)

    recreate_collection(
        client=client,
        collection_name=collection_name,
        vector_size=EMBEDDING_VECTOR_SIZE,
        recreate=recreate,
    )

    embedding_tool = EmbeddingTool(
        model_name=EMBEDDING_MODEL_NAME,
    )
    print(f"[INFO] Embedding runtime selezionato: {embedding_tool.describe_runtime()}")

    try:
        point_id = 0

        batches = list(batched(chunks, EMBEDDING_BATCH_SIZE))

        for batch in tqdm(batches, desc=f"Indexing {collection_name}"):
            texts = [chunk["text"] for chunk in batch]

            vectors = embedding_tool.embed_texts(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
            )

            points = []

            for chunk, vector in zip(batch, vectors):
                if len(vector) != EMBEDDING_VECTOR_SIZE:
                    raise ValueError(
                        f"Dimensione embedding inattesa: {len(vector)}. "
                        f"Attesa: {EMBEDDING_VECTOR_SIZE}. "
                        "Controlla EMBEDDING_VECTOR_SIZE in settings.py."
                    )

                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=build_payload(chunk),
                )

                points.append(point)
                point_id += 1

            client.upsert(
                collection_name=collection_name,
                points=points,
            )
    finally:
        embedding_tool.close()

    print(f"[OK] Indicizzazione completata: {collection_name}")
    print(f"[OK] Qdrant local path: {QDRANT_LOCAL_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        choices=["guidelines", "products", "all"],
        default="all",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target in {"guidelines", "all"}:
        index_chunks(
            chunks_path=GUIDELINE_CHUNKS_PATH,
            collection_name=QDRANT_COLLECTION_GUIDELINES,
            recreate=args.recreate,
        )

    if args.target in {"products", "all"}:
        index_chunks(
            chunks_path=PRODUCT_CHUNKS_PATH,
            collection_name=QDRANT_COLLECTION_PRODUCTS,
            recreate=args.recreate,
        )


if __name__ == "__main__":
    main()

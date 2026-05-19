from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    EMBEDDING_RUNTIME,
    EMBEDDING_TORCH_DEVICE,
    OPENVINO_DEVICE,
    QDRANT_COLLECTION_GUIDELINES,
    QDRANT_COLLECTION_PRODUCTS,
    QDRANT_LOCAL_PATH,
)
from tools.qdrant_client_factory import get_qdrant_client  # noqa: E402
from tools.embedding_tool import EmbeddingTool  # noqa: E402


def make_filter(
    status: Optional[str] = None,
    region: Optional[str] = None,
    document_type: Optional[str] = None,
) -> Optional[Filter]:
    conditions = []

    if status:
        conditions.append(
            FieldCondition(
                key="status",
                match=MatchValue(value=status),
            )
        )

    if region:
        conditions.append(
            FieldCondition(
                key="region",
                match=MatchValue(value=region),
            )
        )

    if document_type:
        conditions.append(
            FieldCondition(
                key="document_type",
                match=MatchValue(value=document_type),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def search(
    collection_name: str,
    query: str,
    top_k: int,
    status: Optional[str] = None,
    region: Optional[str] = None,
    document_type: Optional[str] = None,
) -> None:
    client = get_qdrant_client(QDRANT_LOCAL_PATH)

    embedding_tool = EmbeddingTool(
        model_name=EMBEDDING_MODEL_NAME,
    )

    query_vector = embedding_tool.embed_query(query)

    query_filter = make_filter(
        status=status,
        region=region,
        document_type=document_type,
    )

    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
    except AttributeError:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

    print("\n" + "=" * 100)
    print(f"COLLECTION: {collection_name}")
    print(f"QUERY: {query}")
    print(f"MODEL: {EMBEDDING_MODEL_NAME}")
    print(f"REQUESTED RUNTIME: {EMBEDDING_RUNTIME}")
    print(f"REQUESTED TORCH DEVICE: {EMBEDDING_TORCH_DEVICE}")
    print(f"REQUESTED OPENVINO DEVICE: {OPENVINO_DEVICE}")
    print(f"SELECTED EMBEDDING: {embedding_tool.describe_runtime()}")
    print("=" * 100)

    for i, result in enumerate(results, start=1):
        payload: Dict[str, Any] = result.payload or {}

        print(f"\n--- RESULT {i} ---")
        print(f"score: {result.score:.4f}")
        print(f"chunk_id: {payload.get('chunk_id')}")
        print(f"source_file: {payload.get('source_file')}")

        if collection_name == QDRANT_COLLECTION_PRODUCTS:
            print(f"product_name: {payload.get('product_name')}")
            print(f"registration_number: {payload.get('registration_number')}")
            print(f"status: {payload.get('status')}")
            print(f"active_substances: {payload.get('active_substances')}")
            print(f"categories: {payload.get('categories')}")

        if collection_name == QDRANT_COLLECTION_GUIDELINES:
            print(f"title: {payload.get('title')}")
            print(f"region: {payload.get('region')}")
            print(f"document_type: {payload.get('document_type')}")
            print(f"year: {payload.get('year')}")
            print(f"has_table: {payload.get('has_table')}")
            print(f"topic: {payload.get('topic')}")
            print(f"adversities: {payload.get('adversities')}")
            print(f"rule_types: {payload.get('rule_types')}")

        text = payload.get("text", "")
        print("\nTEXT PREVIEW:")
        print(text[:1200].replace("\n", " "))
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--collection",
        choices=["guidelines", "products"],
        required=True,
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--status",
        default=None,
    )

    parser.add_argument(
        "--region",
        default=None,
    )

    parser.add_argument(
        "--document-type",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    collection_name = (
        QDRANT_COLLECTION_GUIDELINES
        if args.collection == "guidelines"
        else QDRANT_COLLECTION_PRODUCTS
    )

    search(
        collection_name=collection_name,
        query=args.query,
        top_k=args.top_k,
        status=args.status,
        region=args.region,
        document_type=args.document_type,
    )


if __name__ == "__main__":
    main()

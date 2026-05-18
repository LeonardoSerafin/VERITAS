from __future__ import annotations
from pathlib import Path
from qdrant_client import QdrantClient


def get_qdrant_client(local_path: Path) -> QdrantClient:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(local_path))
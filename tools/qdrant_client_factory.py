from __future__ import annotations
from pathlib import Path
from threading import Lock
from qdrant_client import QdrantClient


_clients: dict[Path, QdrantClient] = {}
_clients_lock = Lock()


def get_qdrant_client(local_path: Path) -> QdrantClient:
    resolved_path = local_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with _clients_lock:
        client = _clients.get(resolved_path)
        if client is None:
            client = QdrantClient(
                path=str(resolved_path),
                force_disable_check_same_thread=True,
            )
            _clients[resolved_path] = client

        return client

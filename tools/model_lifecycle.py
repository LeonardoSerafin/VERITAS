from __future__ import annotations

import gc


def release_model_memory() -> None:
    """Best-effort cleanup after large model objects are dereferenced."""
    gc.collect()

    try:
        import torch
    except Exception:
        gc.collect()
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:
        pass

    gc.collect()

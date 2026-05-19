from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import settings  # noqa: E402


def has_openvino_artifacts(output_dir: Path) -> bool:
    direct_xml = output_dir / "openvino_model.xml"
    nested_xml = output_dir / "openvino" / "openvino_model.xml"
    return direct_xml.exists() or nested_xml.exists()


def export_openvino_model(model_id: str, output_dir: Path, device: str, force: bool, verify: bool) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if has_openvino_artifacts(output_dir) and not force:
        print(f"[INFO] Modello reranker OpenVINO gia presente in: {output_dir}")
        print("[INFO] Uso --force per riesportare.")
        return

    print(f"[INFO] Export OpenVINO del reranker: {model_id}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output dir: {output_dir}")

    processor_kwargs = {
        "fix_mistral_regex": True,
    }

    model = CrossEncoder(
        model_id,
        backend="openvino",
        model_kwargs={
            "device": device,
            "export": True,
        },
        processor_kwargs=processor_kwargs,
        prompts={"rag": settings.RERANKER_INSTRUCTION},
        default_prompt_name="rag",
        trust_remote_code=True,
    )

    model.save_pretrained(str(output_dir))

    if verify:
        print("[INFO] Verifica caricamento locale...")
        cache_dir = output_dir / "openvino" / "model_cache"
        loaded_model = CrossEncoder(
            str(output_dir),
            backend="openvino",
            model_kwargs={
                "device": device,
                "export": False,
                "subfolder": "openvino",
                "file_name": "openvino_model.xml",
                "ov_config": {"CACHE_DIR": str(cache_dir)},
            },
            processor_kwargs=processor_kwargs,
            prompts={"rag": settings.RERANKER_INSTRUCTION},
            default_prompt_name="rag",
            local_files_only=True,
            trust_remote_code=True,
        )
        _ = loaded_model.predict(
            [("peronospora vite", "La peronospora della vite richiede interventi preventivi.")],
            batch_size=1,
            show_progress_bar=False,
        )

    print("[OK] Export completato. Il reranker OpenVINO locale e pronto.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esporta una sola volta il reranker in formato OpenVINO locale."
    )
    parser.add_argument(
        "--model-id",
        default=settings.RERANKER_MODEL_NAME,
        help="Model ID Hugging Face sorgente.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(settings.RERANKER_MODEL_LOCAL_PATH),
        help="Directory locale di destinazione per il reranker OpenVINO.",
    )
    parser.add_argument(
        "--device",
        default=settings.RERANKER_OPENVINO_DEVICE,
        help="Device OpenVINO (es: GPU, CPU, AUTO).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forza una nuova esportazione anche se gli artifact sono gia presenti.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verifica immediata del caricamento locale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_openvino_model(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        device=args.device,
        force=args.force,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()

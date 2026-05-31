from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from config import settings


CLASS_TO_IDX = settings.VALIDATOR_CLASS_TO_IDX
VALID_CLASS_INDEX = CLASS_TO_IDX["valid_grape_leaf"]
INVALID_CLASS_INDEX = CLASS_TO_IDX["invalid_image"]

VALID_STATE = "GRAPEVINE_LEAF"
INVALID_STATE = "NOT_GRAPEVINE_LEAF"

DEFAULT_DEVICE = settings.VALIDATOR_DEVICE


class GrapeLeafValidator:
    def __init__(
        self,
        model_path: str | Path = settings.VALIDATOR_MODEL_PATH,
        threshold: float = settings.VALIDATOR_THRESHOLD,
        device: str = settings.VALIDATOR_DEVICE,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.device = self._resolve_device(device)
        self.transform = self._build_transform()
        self.model = self._load_model()

    def validate(self, image: Image.Image) -> tuple[Image.Image, str]:

        """Verifica se l'immagine in input rappresenta una foglia di vite valida."""

        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")

        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        valid_probability = float(probabilities[VALID_CLASS_INDEX])
        invalid_probability = float(probabilities[INVALID_CLASS_INDEX])
        is_valid = valid_probability >= self.threshold

        return image, VALID_STATE if is_valid else INVALID_STATE

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _build_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(settings.VALIDATOR_RESIZE_SIZE),
                transforms.CenterCrop(settings.VALIDATOR_CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(settings.VALIDATOR_IMAGENET_MEAN, settings.VALIDATOR_IMAGENET_STD),
            ]
        )

    @staticmethod
    def _build_model() -> nn.Module:
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(CLASS_TO_IDX))
        return model

    def _load_model(self) -> nn.Module:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Grape leaf validator model not found: {self.model_path}")

        checkpoint = self._load_checkpoint()
        class_to_idx = checkpoint.get("class_to_idx")
        if class_to_idx != CLASS_TO_IDX:
            raise ValueError(
                f"Unexpected checkpoint class mapping: {class_to_idx}. "
                f"Expected: {CLASS_TO_IDX}"
            )

        model = self._build_model()
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _load_checkpoint(self) -> dict[str, Any]:
        try:
            return torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(self.model_path, map_location=self.device)


_validator: GrapeLeafValidator | None = None


def get_grape_leaf_validator() -> GrapeLeafValidator:
    global _validator
    if _validator is None:
        _validator = GrapeLeafValidator()
    return _validator


def validate_grape_leaf(image: Image.Image) -> tuple[Image.Image, str]:

    """Esegue la validazione dell'immagine utilizzando il validatore globale."""

    return get_grape_leaf_validator().validate(image)

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from pytorch_lightning import LightningModule
from config import settings


learning_rate = 1e-3
num_classes = None


class My_CNN_CustomModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.ground_truth = []
        self.predictions = []

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)


class GrapeDiseaseCNNTool:
    """
    Tool riutilizzabile per inferenza su immagini di foglie di vite.

    Input:
        image: oggetto Image.Image (PIL)

    Output:
        dict con malattia predetta, confidence, top-k, device, ecc.
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        image_size: int = 256,
        top_k: int = 4,
        device: Optional[str] = None
    ):
        global num_classes

        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.image_size = image_size
        self.top_k = top_k

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names = settings.CLASS_NAMES
        num_classes = len(self.class_names)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        self.model = My_CNN_CustomModel.load_from_checkpoint(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = F.softmax(logits, dim=1)

            confidence_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1)

            predicted_idx = predicted_idx_tensor.item()
            confidence = confidence_tensor.item()
            predicted_class = self.class_names[predicted_idx]

            top_k = min(self.top_k, len(self.class_names))
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        top_predictions = []

        for rank in range(top_k):
            idx = top_indices[0, rank].item()
            prob = top_probs[0, rank].item()

            top_predictions.append({
                "rank": rank + 1,
                "class_name": self.class_names[idx],
                #"confidence": round(prob, 6),       #da capire se gli LLM preferiscono la confidence in percentuale o in decimale  
                "confidence_percent": round(prob * 100, 2)
            })

        result = {
            "tool_name": "grape_leaf_disease_cnn",
            "predicted_class": predicted_class,
            #"confidence": round(confidence, 6),        #da capire se gli LLM preferiscono la confidence in percentuale o in decimale
            "confidence_percent": round(confidence * 100, 2),
            "top_predictions": top_predictions,
            #"num_classes": len(self.class_names),
            #"class_names": self.class_names,
            #"device": self.device,
            #"status": "success"
        }

        return result


# Istanza globale opzionale, utile se vuoi caricare il modello una volta sola.
_cnn_tool_instance: Optional[GrapeDiseaseCNNTool] = None


def initialize_cnn_tool(
    checkpoint_path: str,
    data_dir: str,
    image_size: int = 256,
    top_k: int = 4,
    device: Optional[str] = None
) -> GrapeDiseaseCNNTool:
    """
    Inizializza il tool CNN una sola volta.
    """
    global _cnn_tool_instance

    _cnn_tool_instance = GrapeDiseaseCNNTool(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        image_size=image_size,
        top_k=top_k,
        device=device
    )

    return _cnn_tool_instance


def analyze_leaf_image(image: Image.Image) -> Dict[str, Any]:
    """
    Funzione-tool da far chiamare agli agenti.

    Esempio:
        result = analyze_leaf_image(image)

    Ritorna:
        {
            "predicted_class": "peronospora",
            "confidence": 0.91,
            ...
        }
    """
    if _cnn_tool_instance is None:
        raise RuntimeError(
            "CNN tool non inizializzato. "
            "Chiama prima initialize_cnn_tool(...)."
        )

    return _cnn_tool_instance.analyze(image)
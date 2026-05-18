from tools.cnn_leaf_disease_tool import initialize_cnn_tool
from config import settings
from masfactory import ImageAsset
from architecture.masfactory_graph import build_architecture
from pprint import pprint

image_path = "dataset/Dataset-splittato/test/ESCA/0b2d8af7-af0b-4192-b60c-5a355b762c65___FAM_B.Msls 4201.JPG"

def main():
    initialize_cnn_tool(
        checkpoint_path=str(settings.MODEL_PATH),
        data_dir=str(settings.DATASET_DIR),
        image_size=settings.IMAGE_SIZE,
        top_k=settings.VISION_TOP_K,
    )

    graph = build_architecture()

    payload = {
        "location": "conegliano",
        "growth_stage": "fioritura",
        "wine_type": "prosecco",
        "recent_treatments": "nessun trattamento recente",
        "image": ImageAsset.from_path(image_path),
    }

    results = graph.invoke(payload)

    pprint(results)


if __name__ == "__main__":
    main()

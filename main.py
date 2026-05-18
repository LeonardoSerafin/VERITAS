from tools.cnn_leaf_disease_tool import initialize_cnn_tool
from config import settings
from masfactory import ImageAsset
from architecture.masfactory_graph import build_architecture
from architecture.live_monitor import install_live_hooks
from pprint import pprint

image_path = "example_dataset/Black Rot/0aff8add-93ad-4099-97ae-23515744e620___FAM_B.Rot 0748.JPG"

def main():
    initialize_cnn_tool(
        checkpoint_path=str(settings.MODEL_PATH),
        data_dir=str(settings.DATASET_DIR),
        image_size=settings.IMAGE_SIZE,
        top_k=settings.VISION_TOP_K,
    )

    graph = build_architecture()
    install_live_hooks(graph)

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

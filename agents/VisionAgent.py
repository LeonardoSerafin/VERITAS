import os
from masfactory import CustomNode, RootGraph, ImageAsset
from tempfile import NamedTemporaryFile
from tools.cnn_leaf_disease_tool import analyze_leaf_image
from PIL import Image
from pathlib import Path
import io


def ImageAsset_to_PIL_Image(image_asset: ImageAsset) -> Image.Image:
    # Convert the ImageAsset to bytes
    image_bytes = image_asset.load_bytes()   
    # Create a BytesIO object from the bytes
    image_stream = io.BytesIO(image_bytes)
    # Open the image using PIL
    pil_image = Image.open(image_stream)

    return pil_image


# VisionAgent function that will be used as the forward function for the CustomNode in the MASFactory graph
def vision_agent_function(input_data: dict) -> dict:
    image = input_data["image"]
    if not image:
        raise ValueError("Input data must contain 'image' key.")

    PIL_image = ImageAsset_to_PIL_Image(image)

    cnn_result = analyze_leaf_image(PIL_image)

    agent_output = {
        "agent_name": "VisionAgent",
        "task": "leaf_disease_classification",
        "disease": cnn_result["predicted_class"],
        "confidence_percent": cnn_result["confidence_percent"],
        "top_predictions": cnn_result["top_predictions"],
    }

    return agent_output


def create_vision_agent_node(graph: RootGraph, node_name: str = "VisionAgentNode"):    
    return graph.create_node(
        CustomNode,
        name=node_name,
        forward=vision_agent_function,
    )
    
import os
from masfactory import CustomNode, RootGraph, ImageAsset
from tempfile import NamedTemporaryFile
from tools.cnn_leaf_disease_tool import analyze_leaf_image
from PIL import Image
from pathlib import Path


# Function to convert image input to a temporary file path since the CNN tool expects a file path as input
def image_to_temp_path(image_input) -> str:
    
    if isinstance(image_input, (str, Path)) and os.path.isfile(image_input):
        return image_input
    
    if isinstance(image_input, Image.Image):
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.close()
        image_input.convert("RGB").save(temp_file.name, format="JPEG")
        return temp_file.name
    
    # Shitty solution to handle the case when the input is an ImageAsset from MASFactory, which doesn't have a direct method to get the file path. We save it to a temporary file and return that path.
    if isinstance(image_input, ImageAsset):
        image_input_bytes = image_input.load_bytes()
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        with temp_file:
            temp_file.write(image_input_bytes)
        temp_file.close()
        return temp_file.name

    raise ValueError("Unsupported image input type. Expected file path or PIL Image object.")


# VisionAgent function that will be used as the forward function for the CustomNode in the MASFactory graph
def vision_agent_function(input_data: dict) -> dict:
    image = input_data["image"]
    if not image:
        raise ValueError("Input data must contain 'image' key.")
    
    # Capiamo se il file era già un path esistente all'inizio
    is_temp_file = not (isinstance(image, (str, Path)) and os.path.isfile(image))
    
    image_path = image_to_temp_path(image)

    cnn_result = analyze_leaf_image(image_path)

    if is_temp_file:
        os.remove(image_path)

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
    
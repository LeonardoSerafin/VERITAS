from masfactory import CustomNode, RootGraph

def bypass_node_function(input_data: dict) -> dict:
    return {
        "rag_context": ""
    }

def create_bypass_node(graph: RootGraph, node_name: str = "BypassNode"):
    return graph.create_node(
        CustomNode,
        name=node_name,
        forward=bypass_node_function
    )
from masfactory import LogicSwitch, RootGraph

def is_not_healthy(msg: dict, _attrs: dict) -> bool:
    disease = str(msg.get("disease", "")).strip().lower()
    return disease != "healthy"

def is_healthy(msg: dict, _attrs: dict) -> bool:
    disease = str(msg.get("disease", "")).strip().lower()
    return disease == "healthy"

def create_conditional_node(graph: RootGraph, node_name: str = "ConditionalNode"):
    return graph.create_node(
        LogicSwitch,
        name=node_name,
        routes={
            "RAGAgentNode": is_not_healthy,
            "BypassNode": is_healthy,
        },
    )
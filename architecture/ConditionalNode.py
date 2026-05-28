from masfactory import LogicSwitch, RootGraph

def rag_needed(msg: dict, _attrs: dict) -> bool:
    disease = str(msg.get("disease", "")).strip().lower()
    state = str(msg.get("state", "")).strip().lower() 
    return (disease != "sana") and (state == "valid image")

def rag_not_needed(msg: dict, _attrs: dict) -> bool:
    disease = str(msg.get("disease", "")).strip().lower()
    state = str(msg.get("state", "")).strip().lower()
    return (disease == "sana") or (state == "invalid image")

def create_conditional_node(graph: RootGraph, node_name: str = "ConditionalNode"):
    return graph.create_node(
        LogicSwitch,
        name=node_name,
        routes={
            "RAGAgentNode": rag_needed,
            "BypassNode": rag_not_needed,
        },
    )
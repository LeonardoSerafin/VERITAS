from agents.ContextAgent import create_context_agent_node
from agents.DecisionAgent import create_decision_agent_node
from agents.InputParser import create_input_parser_node
from agents.RAGAgent import create_rag_agent_node
from agents.VisionAgent import create_vision_agent_node
from architecture.BypassNode import create_bypass_node
from architecture.ConditionalNode import create_conditional_node
from masfactory import RootGraph


def build_architecture():
    """
    Build and return a full-system MASFactory graph with fixed topology.
    """

    graph = RootGraph(name="VERITAS")

    inputParser = create_input_parser_node(graph)
    visionAgent = create_vision_agent_node(graph)
    contextAgent = create_context_agent_node(graph)
    conditionalNode = create_conditional_node(graph)
    ragAgent = create_rag_agent_node(graph)
    bypassNode = create_bypass_node(graph)
    decisionAgent = create_decision_agent_node(graph)

    graph.edge_from_entry(
        inputParser,
        {
            "location": "location of the vineyard",
            "growth_stage": "current growth stage of the vine",
            "wine_type": "type of wine produced by the vineyard",
            "recent_treatments": "recent treatments applied to the vineyard",
            "image": "IMAGE:Leaf image input for disease classification",
        },
    )

    graph.create_edge(
        inputParser,
        visionAgent,
        {"image": "IMAGE:Leaf image input for disease classification"},
    )

    graph.create_edge(
        inputParser,
        contextAgent,
        {
            "location": "location of the vineyard",
            "growth_stage": "current growth stage of the vine",
            "wine_type": "type of wine produced by the vineyard",
            "recent_treatments": "recent treatments applied to the vineyard",
        },
    )

    graph.create_edge(
        inputParser,
        conditionalNode,
        {"location": "location of the vineyard"},
    )

    graph.create_edge(
        visionAgent,
        conditionalNode,
        {
            "disease": "predicted disease",
            "confidence_percent": "confidence_percent",
            "top_predictions": "top_predictions",
        },
    )

    graph.create_edge(
        conditionalNode,
        ragAgent,
        {
            "disease": "predicted disease",
            "confidence_percent": "confidence_percent",
            "top_predictions": "top_predictions",
            "location": "location of the vineyard",
        },
    )

    graph.create_edge(
        conditionalNode,
        bypassNode,
        {
            "disease": "predicted disease",     # Tecnchically useless, since the bypass node doesn't use it
        },
    )

    graph.create_edge(
            ragAgent,
            decisionAgent,
            {
                "rag_context": "combined retrieval context from guidelines and authorized products",
                "guidelines_hits": "top guideline chunks from vector search",
                "rag_query": "query generated for vector retrieval",
            },
        )
    
    graph.create_edge(
        bypassNode,
        decisionAgent,
        {
            "rag_context": "empty context to signal no info is needed",
        },
    )
    
    graph.create_edge(
        contextAgent,
        decisionAgent,
        {
            "location": "location of the vineyard",
            "growth_stage": "current growth stage of the vine",
            "wine_type": "type of wine produced by the vineyard",
            "recent_treatments": "recent treatments applied to the vineyard",
            "meteo_forecast": "weather forecast for the next days",
        },
    )

    graph.create_edge(
        visionAgent,
        decisionAgent,
        {
            "disease": "predicted disease",
            "confidence_percent": "confidence_percent",
            "top_predictions": "top_predictions",
        },
    )

    graph.edge_to_exit(
        decisionAgent,
        {
            "predicted_disease": "final predicted disease based on the integrated analysis",
            "risk_level": "estimated risk level for the vineyard (LOW, MEDIUM, HIGH)",
            "decision_report": "final decision support recommendation with rationale and uncertainty",
        },
    )

    graph.build()
    return graph

from masfactory import Agent, NodeTemplate, RootGraph
from config import settings
from tools.weather_tool import get_weather_forecast, initialize_weather_tool

initialize_weather_tool(default_source="dwd-icon")

ContextAgentTemplate = NodeTemplate(
    Agent,
    model = settings.DEFAULT_LLM_MODEL,
    instructions = (
        "You are a helpful assistant for farmers to identify diseases in their grapevines."
        "Your job is to prepare a rich context for the Decision Agent in order to have a more precise diagnosis of the disease affecting the vine."
        "Use get_weather_forecast tool to get the weather forecast for the next days"
    ),
    prompt_template = (
        "[LOCATION] \nThe vineyard is located in {location}.\n\n "
        "[GROWTH_STAGE] \nThe vine is currently in the {growth_stage} stage.\n\n"
        "[WINE TYPE] \nThe vineyard produces {wine_type} wine.\n\n"
        "[RECENT TREATMENTS] \nThe recent treatments applied to the vineyard are {recent_treatments}.\n\n"
    ),
    tools = [get_weather_forecast]
)

def create_context_agent_node(graph: RootGraph, node_name: str = "ContextAgentNode"):
    return graph.create_node(
        ContextAgentTemplate,
        name=node_name,
    )
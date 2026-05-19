from masfactory import Agent, NodeTemplate, RootGraph
from config import settings
from tools.weather_tool import get_weather_forecast, initialize_weather_tool

initialize_weather_tool(default_source="dwd-icon")

ContextAgentTemplate = NodeTemplate(
    Agent,
    model = settings.DEFAULT_LLM_MODEL,
    instructions = (
        "Sei un assistente utile per gli agricoltori per identificare le malattie nelle loro viti."
        "Il tuo compito è preparare un contesto ricco per il Decision Agent al fine di avere una diagnosi più precisa della malattia che colpisce la vite."
        "Utilizza lo strumento get_weather_forecast per ottenere le previsioni meteo per i prossimi giorni"
        "Il report meteo che fornisci deve essere dettagliato e specifico per la località del vigneto, includendo informazioni su temperatura, umidità, precipitazioni, vento per ogni singolo giorno."
        "Non fare nessun tipo di assunzione su malattie, fornisci solo informazioni contestuali che possano essere utili al Decision Agent per la diagnosi e la gestione della malattia."
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

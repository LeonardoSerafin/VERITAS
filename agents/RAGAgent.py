from masfactory import Agent, NodeTemplate, RootGraph, Factory
from tools.rag_retrieval_tool import QdrantRetrieval
from config import settings

def create_guidelines_retriever():
    return QdrantRetrieval(
        collection_name=settings.QDRANT_COLLECTION_GUIDELINES,
        context_label="GUIDELINES_QDRANT",
        passive=True,
        active=False,
        default_top_k=settings.RAG_TOP_K_GUIDELINES,
        candidate_top_k=settings.RAG_CANDIDATE_TOP_K_GUIDELINES,
    )

RAGAgentTemplate = NodeTemplate(
    Agent,
    model = settings.DEFAULT_LLM_MODEL,
    instructions = (
        "Sei il RAGAgent di un sistema multi-agente per supporto decisionale in viticoltura."
        "Ricevi in input posizione del vigneto, malattia predetta, confidence score e top alternative diagnostiche."
        "Il tuo compito è cercare nelle knowledge base le informazioni più rilevanti su malattia, sintomi, condizioni favorevoli, fase fenologica, monitoraggio, difesa, vincoli e prodotti/sostanze eventualmente citati."
        "Dai priorità a fonti coerenti con la regione del vigneto e considera le alternative diagnostiche se la confidence è bassa o incerta."
        "Non prendere decisioni finali e non prescrivere trattamenti: fornisci solo evidenze documentali utili al DecisionAgent."
        "Cita sempre le fonti recuperate indicando chunk_id, documento, pagina/sezione e score quando disponibili."
        "Se i dati sui prodotti non confermano l'impiego autorizzato specifico su vite/avversità, segnalalo esplicitamente."
        "Non inventare informazioni mancanti: evidenzia incertezze, limiti delle fonti e contesto non disponibile."
    ),
    prompt_template = (
        "[LOCATION]: \nThe vineyard is located in {location}.\n\n "
        "[DISEASE]: \nThe predicted disease by the cnn is {disease}.\n"
        "[CONFIDENCE_PERCENT]: \nThe confidence score of the prediction is {confidence_percent}.\n"
        "[TOP_PREDICTIONS]: \nThe alternative diagnoses are: {top_predictions}.\n\n"
     ),
     retrievers = [Factory(create_guidelines_retriever)]
)

def create_rag_agent_node(graph: RootGraph, node_name: str = "RAGAgentNode"):
    return graph.create_node(
        RAGAgentTemplate,
        name=node_name,
    )
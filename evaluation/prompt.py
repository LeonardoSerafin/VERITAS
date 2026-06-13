"""
Rubric and prompt builder for the offline LLM evaluator.

The evaluator judges a single VERITAS run (one JSON produced by run_cases.py).
The rubric is *adaptive to the image state* captured by the VisionAgent:

- "valid image"   -> full agronomic rubric (the system is expected to diagnose
                     and recommend).
- "invalid image" / "unusable image" -> rejection rubric (the system is expected
                     to refuse, explain, and ask for a new photo; agronomic and
                     RAG criteria do not apply).

This module is pure: it only reads a run record (dict) and returns strings /
structured rubric metadata. The actual LLM call lives in evaluate_outputs.py.
"""

from __future__ import annotations

import json
from typing import Any


# --------------------------------------------------------------------------- #
# Rubrics
# --------------------------------------------------------------------------- #
# Each rubric maps a stable criterion key (used in the output JSON) to a short
# human description shown to the evaluator. Keys are snake_case and language
# agnostic; descriptions are in Italian to match the domain.

RUBRIC_VALID: dict[str, str] = {
    "coerenza_input_visivo": (
        "La raccomandazione rispetta la diagnosi e lo state del VisionAgent "
        "(malattia, top_predictions) senza contraddirli o ignorarli?"
    ),
    "uso_confidenza": (
        "La confidenza della CNN e' usata correttamente? Bassa confidenza deve "
        "tradursi in cautela e trattamento della diagnosi come ipotesi, non come "
        "fatto certo."
    ),
    "coerenza_agronomica": (
        "Le raccomandazioni sono plausibili per fase fenologica, meteo, localita' "
        "e tipo di vino indicati nel contesto?"
    ),
    "uso_corretto_rag": (
        "Il report usa le evidenze recuperate (FONTI RECUPERATE + rag_context) "
        "senza inventare informazioni non presenti? Rispetta i vincoli/regole "
        "recuperati e non li contraddice?"
    ),
    "citazioni_tracciabilita": (
        "Le fonti sono citate vicino all'affermazione che supportano (non solo alla "
        "fine)? Una citazione e' tracciabile se il documento e/o la pagina/sezione/"
        "chunk compaiono OVUNQUE nelle FONTI RECUPERATE (sotto qualsiasi nome di "
        "campo): NON considerarla inventata se il riferimento e' presente li', anche "
        "quando non e' nella prosa del rag_context."
    ),
    "sicurezza_decisionale": (
        "Il sistema evita trattamenti aggressivi quando i dati sono insufficienti, "
        "ambigui o contraddittori, indicando comunque come ridurre l'incertezza?"
    ),
    "completezza_operativa": (
        "Sono presenti azioni pratiche, priorita', monitoraggio, follow-up e "
        "condizioni che farebbero cambiare decisione?"
    ),
    "non_allucinazione": (
        "Il report NON introduce prodotti, dosi, tempi di carenza, documenti, pagine "
        "o vincoli legali non supportati dagli input o dalle evidenze recuperate "
        "(FONTI RECUPERATE + rag_context)? Un documento/pagina/sezione citato che "
        "compare OVUNQUE nelle FONTI RECUPERATE (sotto qualsiasi nome di campo) NON "
        "e' un'allucinazione; lo e' solo se non e' rintracciabile li'."
    ),
    "chiarezza": (
        "Il report e' comprensibile e ben strutturato per un utente agricolo/tecnico?"
    ),
    "utilita_finale": (
        "Nel complesso, il report aiuta a prendere una decisione responsabile e "
        "concretamente utile?"
    ),
}

RUBRIC_INVALID: dict[str, str] = {
    "rifiuto_corretto": (
        "Il sistema riconosce che l'immagine non e' utilizzabile (state 'invalid "
        "image' o 'unusable image') e NON produce diagnosi o raccomandazioni "
        "agronomiche?"
    ),
    "richiesta_nuova_foto": (
        "Il sistema chiede esplicitamente all'utente di fornire una nuova immagine?"
    ),
    "assenza_allucinazioni": (
        "Il sistema NON inventa una malattia, un trattamento o un contesto "
        "nonostante l'input non sia analizzabile?"
    ),
    "chiarezza": (
        "Il messaggio di rifiuto e' chiaro e comprensibile, spiegando perche' non "
        "si puo' procedere?"
    ),
}

VALID_STATES = {"valid image"}
INVALID_STATES = {"invalid image", "unusable image"}


def get_rubric_for_state(state: str | None) -> tuple[str, dict[str, str]]:
    """Return (mode_label, rubric) for the given VisionAgent state."""
    normalized = (state or "").strip().lower()
    if normalized in INVALID_STATES:
        return "rejection", RUBRIC_INVALID
    # Default to the full rubric for valid images (and any unexpected state, so
    # the run is still scored rather than silently skipped).
    return "agronomic", RUBRIC_VALID


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #

EVALUATOR_SYSTEM_PROMPT = (
    "Sei un valutatore esperto di sistemi di supporto decisionale in viticoltura. "
    "Il tuo compito e' giudicare, in modo rigoroso e imparziale, l'output prodotto "
    "da VERITAS (un sistema multi-agente) per un singolo caso. "
    "Valuti SOLO sulla base delle informazioni fornite nel caso: non aggiungere "
    "conoscenza esterna sulla malattia e non penalizzare il sistema per cose che "
    "non poteva sapere dagli input ricevuti. "
    "Sei severo sulle allucinazioni: se il sistema cita prodotti, dosi o vincoli "
    "non presenti nel rag_context o negli input, e' un problema grave. "
    "Assegni a ciascun criterio un punteggio intero da 1 (pessimo) a 5 (ottimo), "
    "con una breve motivazione concreta che faccia riferimento al testo valutato. "
    "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, senza testo prima o dopo, "
    "senza blocchi di codice markdown."
)


# --------------------------------------------------------------------------- #
# Run record -> readable summary
# --------------------------------------------------------------------------- #

def _node_output(run: dict[str, Any], node_name: str) -> dict[str, Any]:
    """Return the recorded output dict of a node, or {} if absent/non-dict."""
    node = run.get("nodes", {}).get(node_name, {})
    output = node.get("output")
    return output if isinstance(output, dict) else {}


def extract_state(run: dict[str, Any]) -> str | None:
    """Read the VisionAgent state from a run record."""
    return _node_output(run, "VisionAgentNode").get("state")


def build_run_summary(run: dict[str, Any]) -> str:
    """
    Build a compact, readable description of what the system received and
    produced, used as the evaluation target. Reads structured node outputs
    rather than the ambiguous final_output list.
    """
    payload = run.get("input", {}) or {}
    vision = _node_output(run, "VisionAgentNode")
    context = _node_output(run, "ContextAgentNode")
    rag = _node_output(run, "RAGAgentNode")
    decision = _node_output(run, "DecisionAgentNode")

    rag_context = rag.get("rag_context", "")
    if not rag_context:
        rag_context = "(nessun RAG: ramo bypass o non eseguito)"

    # Retrieval hits carry the verifiable citation metadata (document, page/
    # section, chunk_id, score). The RAG agent emits them with an inconsistent
    # schema: list of dicts, list of strings, or a single prose string, and the
    # page can live under different keys (page, page_section, pagina_sezione,
    # sezione...). So we dump the hits VERBATIM instead of cherry-picking keys:
    # the evaluator must see exactly the evidence the DecisionAgent received and
    # match citations against it, whatever its shape.
    hits = rag.get("guidelines_hits")
    if hits:
        if isinstance(hits, str):
            hits_block = hits
        else:
            hits_block = json.dumps(hits, ensure_ascii=False, indent=2)
    else:
        hits_block = "(nessuna fonte: ramo bypass o non eseguito)"

    lines = [
        "## INPUT UTENTE",
        f"- Localita': {payload.get('location', '')}",
        f"- Fase fenologica: {payload.get('growth_stage', '')}",
        f"- Tipo di vino: {payload.get('wine_type', '')}",
        f"- Trattamenti recenti: {payload.get('recent_treatments', '')}",
        "",
        "## VISION AGENT",
        f"- State immagine: {vision.get('state', '')}",
        f"- Malattia predetta: {vision.get('disease', '')}",
        f"- Confidenza: {vision.get('confidence_percent', '')}",
        f"- Top predictions: {vision.get('top_predictions', '')}",
        "",
        "## CONTEXT AGENT",
        f"- Meteo: {context.get('meteo_forecast', '')}",
        "",
        "## FONTI RECUPERATE (citazioni verificabili — guidelines_hits, verbatim)",
        "Evidenza grezza ricevuta dal DecisionAgent. Una citazione e' valida se "
        "il documento e/o la pagina/sezione/chunk compaiono OVUNQUE qui sotto, "
        "sotto qualsiasi nome di campo (page, page_section, pagina_sezione, "
        "sezione...). NON considerarla inventata se il riferimento e' presente.",
        hits_block,
        "",
        "## RAG CONTEXT (sintesi in prosa del RAG agent)",
        str(rag_context),
        "",
        "## OUTPUT DECISION AGENT (oggetto da valutare)",
        f"- predicted_disease: {decision.get('predicted_disease', '')}",
        f"- risk_level: {decision.get('risk_level', '')}",
        "- decision_report:",
        str(decision.get("decision_report", "")),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Full evaluation prompt (user message)
# --------------------------------------------------------------------------- #

def _format_rubric(rubric: dict[str, str]) -> str:
    return "\n".join(f'- "{key}": {desc}' for key, desc in rubric.items())


def _format_output_schema(rubric: dict[str, str]) -> str:
    score_fields = ",\n".join(
        f'    "{key}": {{"score": <1-5>, "rationale": "<breve motivazione>"}}'
        for key in rubric
    )
    return (
        "{\n"
        f'  "scores": {{\n{score_fields}\n  }},\n'
        '  "overall_score": <numero 1-5, media ragionata dei criteri>,\n'
        '  "summary": "<2-3 frasi di giudizio complessivo>",\n'
        '  "red_flags": ["<eventuali problemi gravi, lista vuota se nessuno>"]\n'
        "}"
    )


def build_evaluation_prompt(run: dict[str, Any]) -> tuple[str, str]:
    """
    Build the evaluator user message for a run record.

    Returns (mode_label, user_prompt). mode_label is "agronomic" or "rejection"
    so the caller can record which rubric was applied.
    """
    state = extract_state(run)
    mode_label, rubric = get_rubric_for_state(state)

    case = run.get("case", {})
    header = (
        f"Caso: {case.get('id', 'sconosciuto')} "
        f"(state immagine: {state or 'sconosciuto'}; rubrica: {mode_label})"
    )

    prompt = (
        f"{header}\n\n"
        f"{build_run_summary(run)}\n\n"
        "---\n"
        "## RUBRICA DI VALUTAZIONE\n"
        "Assegna a OGNI criterio un punteggio intero da 1 a 5.\n"
        f"{_format_rubric(rubric)}\n\n"
        "## FORMATO DI OUTPUT (JSON, nient'altro)\n"
        f"{_format_output_schema(rubric)}\n"
    )
    return mode_label, prompt


# Convenience for callers that want the criterion keys without building a prompt.
def rubric_keys_for_state(state: str | None) -> list[str]:
    _, rubric = get_rubric_for_state(state)
    return list(rubric.keys())


if __name__ == "__main__":
    # Tiny smoke test against a run file passed as argv[1].
    import sys

    if len(sys.argv) > 1:
        run_record = json.load(open(sys.argv[1], encoding="utf-8"))
        mode, user_prompt = build_evaluation_prompt(run_record)
        print(f"[mode={mode}]\n")
        print(user_prompt)
    else:
        print("Usage: python evaluation/prompt.py <run.json>")

from __future__ import annotations

import html
import time
import traceback
from typing import Any

import streamlit as st


APP_TITLE = "VERITAS"
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

GROWTH_STAGE_OPTIONS = [
    "Germogliamento",
    "Prefioritura",
    "Fioritura",
    "Allegagione",
    "Invaiatura",
    "Maturazione",
    "Post-raccolta",
]

NODE_ORDER = [
    "VisionAgentNode",
    "ContextAgentNode",
    "RAGAgentNode",
    "DecisionAgentNode",
]

NODE_LABELS = {
    "VisionAgentNode": "VisionAgent",
    "ContextAgentNode": "ContextAgent",
    "RAGAgentNode": "RAGAgent",
    "DecisionAgentNode": "DecisionAgent",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(180deg, #f9fbf5 0%, #f4f8ee 100%);
            }
            .veritas-title {
                background: linear-gradient(135deg, #2f6f3e 0%, #4b8f52 100%);
                color: #ffffff;
                border-radius: 14px;
                padding: 18px 20px;
                margin-bottom: 8px;
                box-shadow: 0 8px 24px rgba(36, 84, 47, 0.20);
            }
            .veritas-subtitle {
                color: #34533b;
                margin-top: 2px;
                margin-bottom: 14px;
            }
            div[data-testid="stCheckbox"] label {
                background: var(--secondary-background-color);
                border: 1px solid rgba(127, 127, 127, 0.28);
                border-radius: 10px;
                padding: 8px 10px;
            }
            div[data-testid="stCheckbox"] label p {
                color: var(--text-color) !important;
            }
            div[data-testid="stCheckbox"] input[type="checkbox"] {
                accent-color: #2f6f3e;
                transform: scale(1.15);
            }
            .status-card {
                background: var(--secondary-background-color);
                border: 1px solid rgba(127, 127, 127, 0.28);
                color: var(--text-color);
                border-radius: 12px;
                padding: 10px 12px;
                margin-bottom: 8px;
            }
            .status-line {
                margin: 5px 0;
                font-size: 0.97rem;
            }
            .report-card {
                background: var(--secondary-background-color);
                border: 1px solid rgba(127, 127, 127, 0.28);
                color: var(--text-color);
                border-radius: 14px;
                padding: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def initialize_runtime() -> dict[str, Any]:
    deps = load_runtime_dependencies()
    settings = deps["settings"]

    deps["initialize_cnn_tool"](
        checkpoint_path=str(settings.MODEL_PATH),
        data_dir=str(settings.DATASET_DIR),
        image_size=settings.IMAGE_SIZE,
        top_k=settings.VISION_TOP_K,
    )
    return deps


def render_agent_status(slot: Any, agent_state: dict[str, dict[str, Any]]) -> None:
    if not agent_state:
        slot.info("In attesa di invio input.")
        return

    lines: list[str] = []
    for node_name in NODE_ORDER:
        if node_name not in agent_state:
            continue
        row = agent_state[node_name]
        label = NODE_LABELS[node_name]
        status = row.get("status")
        elapsed_ms = row.get("elapsed_ms")

        if status == "running":
            lines.append(f'<div class="status-line">[RUN] <strong>{label}</strong>: in lavorazione...</div>')
        elif status == "done":
            if elapsed_ms is None:
                lines.append(f'<div class="status-line">[OK] <strong>{label}</strong>: completato</div>')
            else:
                lines.append(
                    f'<div class="status-line">[OK] <strong>{label}</strong>: completato in {elapsed_ms:.1f} ms</div>'
                )
        elif status == "error":
            lines.append(f'<div class="status-line">[ERR] <strong>{label}</strong>: errore</div>')

    if not lines:
        slot.info("In attesa di avvio agenti.")
        return

    slot.markdown(
        '<div class="status-card"><strong>Stato elaborazione</strong><br>' + "".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


def install_agent_progress_hooks(graph: Any, node_cls: Any, status_slot: Any) -> None:
    tracked_nodes = set(NODE_ORDER)
    agent_state: dict[str, dict[str, Any]] = {}
    started_at: dict[str, float] = {}

    def _is_target(node_obj: Any) -> bool:
        return str(getattr(node_obj, "name", "")).strip() in tracked_nodes

    def _render() -> None:
        render_agent_status(status_slot, agent_state)

    def on_execute_before(node: Any, *_args: Any, **_kwargs: Any) -> None:
        node_name = str(getattr(node, "name", "")).strip()
        if node_name not in tracked_nodes:
            return
        started_at[node_name] = time.perf_counter()
        agent_state[node_name] = {"status": "running", "elapsed_ms": None}
        _render()

    def on_execute_after(node: Any, _result: Any, *_args: Any, **_kwargs: Any) -> None:
        node_name = str(getattr(node, "name", "")).strip()
        if node_name not in tracked_nodes:
            return
        start = started_at.pop(node_name, None)
        elapsed_ms = None if start is None else (time.perf_counter() - start) * 1000.0
        agent_state[node_name] = {"status": "done", "elapsed_ms": elapsed_ms}
        _render()

    def on_forward_error(node: Any, _err: Exception, *_args: Any, **_kwargs: Any) -> None:
        node_name = str(getattr(node, "name", "")).strip()
        if node_name not in tracked_nodes:
            return
        agent_state[node_name] = {"status": "error", "elapsed_ms": None}
        _render()

    graph.hook_register(
        node_cls.Hook.EXECUTE.BEFORE,
        on_execute_before,
        recursion=True,
        target_type=node_cls,
        target_filter=_is_target,
    )
    graph.hook_register(
        node_cls.Hook.EXECUTE.AFTER,
        on_execute_after,
        recursion=True,
        target_type=node_cls,
        target_filter=_is_target,
    )
    graph.hook_register(
        node_cls.Hook.FORWARD.ERROR,
        on_forward_error,
        recursion=True,
        target_type=node_cls,
        target_filter=_is_target,
    )


def ensure_treatment_rows() -> None:
    if "treatment_rows" not in st.session_state:
        st.session_state.treatment_rows = [{"name": "", "days": None}]


def render_treatments_input(disabled: bool) -> list[dict[str, Any]]:
    ensure_treatment_rows()
    rows = st.session_state.treatment_rows
    updated_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        name_key = f"treatment_name_{idx}"
        days_key = f"treatment_days_{idx}"

        if name_key not in st.session_state:
            st.session_state[name_key] = row.get("name", "")
        if days_key not in st.session_state:
            st.session_state[days_key] = row.get("days", None)

        col_name, col_days = st.columns([3, 1])
        with col_name:
            name = st.text_input(
                f"Trattamento {idx + 1}",
                key=name_key,
                placeholder="Es. rame, zolfo, fosfonato...",
                disabled=disabled,
            )
        with col_days:
            days = st.number_input(
                "Giorni fa",
                key=days_key,
                min_value=0,
                step=1,
                value=st.session_state[days_key],
                placeholder="Es. 7",
                disabled=disabled,
            )

        updated_rows.append(
            {
                "name": (name or "").strip(),
                "days": days,
            }
        )

    st.session_state.treatment_rows = updated_rows

    if not disabled and updated_rows:
        last_row = updated_rows[-1]
        if last_row["name"] and last_row["days"] is not None:
            st.session_state.treatment_rows.append({"name": "", "days": None})
            st.rerun()

    if not disabled and len(updated_rows) > 1:
        if st.button("Rimuovi ultimo trattamento", use_container_width=True):
            last_idx = len(updated_rows) - 1
            st.session_state.pop(f"treatment_name_{last_idx}", None)
            st.session_state.pop(f"treatment_days_{last_idx}", None)
            st.session_state.treatment_rows = updated_rows[:-1]
            st.rerun()

    return updated_rows


def format_recent_treatments(no_recent: bool, rows: list[dict[str, Any]]) -> str:
    if no_recent:
        return "nessun trattamento recente"

    chunks: list[str] = []
    for row in rows:
        name = (row.get("name") or "").strip()
        days = row.get("days")
        if not name:
            continue
        if days is None:
            chunks.append(name)
        else:
            chunks.append(f"{name} ({int(days)} giorni fa)")
    return "; ".join(chunks)


def parse_graph_output(raw_result: Any) -> dict[str, Any]:
    if isinstance(raw_result, tuple) and len(raw_result) >= 1:
        output = raw_result[0]
    else:
        output = raw_result

    if not isinstance(output, dict):
        raise ValueError("Output grafo inatteso: atteso dict.")
    return output


def validate_inputs(
    uploaded_file: Any,
    location: str,
    wine_type: str,
    no_recent_treatments: bool,
    treatment_rows: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    if uploaded_file is None:
        errors.append("Carica un'immagine della foglia.")
    elif getattr(uploaded_file, "size", 0) > MAX_UPLOAD_BYTES:
        errors.append(f"L'immagine supera {MAX_UPLOAD_MB} MB.")

    if not location.strip():
        errors.append("Inserisci la localita.")

    if not wine_type.strip():
        errors.append("Inserisci la tipologia di vino.")

    if not no_recent_treatments:
        non_empty_rows = [r for r in treatment_rows if (r.get("name") or "").strip()]
        if not non_empty_rows:
            errors.append("Inserisci almeno un trattamento recente oppure seleziona 'Nessun trattamento recente'.")
        else:
            for i, row in enumerate(non_empty_rows, start=1):
                if row.get("days") is None:
                    errors.append(f"Specifica i giorni per il trattamento {i}.")

    return errors


def load_runtime_dependencies() -> dict[str, Any]:
    """
    Lazy imports so the app can render diagnostics even if environment is incomplete.
    """
    from masfactory import ImageAsset, Node  # type: ignore
    from architecture.masfactory_graph import build_architecture  # type: ignore
    from config import settings  # type: ignore
    from tools.cnn_leaf_disease_tool import initialize_cnn_tool  # type: ignore

    return {
        "ImageAsset": ImageAsset,
        "Node": Node,
        "build_architecture": build_architecture,
        "settings": settings,
        "initialize_cnn_tool": initialize_cnn_tool,
    }


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()

    st.markdown(
        f'<div class="veritas-title"><h1 style="margin:0;">{APP_TITLE}</h1></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="veritas-subtitle">Supporto decisionale in viticoltura con pipeline multi-agente</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.subheader("Input")
        uploaded_file = st.file_uploader(
            "Carica immagine foglia",
            type=["jpg", "jpeg", "png"],
            help=f"Formati supportati: JPG/JPEG/PNG. Max {MAX_UPLOAD_MB} MB.",
        )
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Anteprima immagine", use_container_width=True)

    with col_right:
        st.subheader("Dati vigneto")
        location = st.text_input("Localita", placeholder="Es. Conegliano")
        wine_type = st.text_input("Tipologia di vino", placeholder="Es. Prosecco")
        growth_stage = st.selectbox("Fase fenologica", options=GROWTH_STAGE_OPTIONS, index=2)

        st.markdown("**Trattamenti recenti**")
        no_recent_treatments = st.checkbox("Nessun trattamento recente", value=False)
        treatment_rows = render_treatments_input(disabled=no_recent_treatments)

    submit_slot = st.empty()
    submit = submit_slot.button("Invia analisi", type="primary", use_container_width=True)
    status_slot = st.empty()

    if submit:
        submit_slot.empty()

        errors = validate_inputs(
            uploaded_file=uploaded_file,
            location=location,
            wine_type=wine_type,
            no_recent_treatments=no_recent_treatments,
            treatment_rows=treatment_rows,
        )
        if errors:
            for err in errors:
                st.error(err)
            return

        recent_treatments = format_recent_treatments(
            no_recent=no_recent_treatments,
            rows=treatment_rows,
        )

        try:
            with st.spinner("Analisi in corso...", show_time=True):
                deps = initialize_runtime()

                image_bytes = uploaded_file.getvalue()
                image_asset = deps["ImageAsset"].from_bytes(
                    image_bytes,
                    mime_type=uploaded_file.type or "image/jpeg",
                    filename=uploaded_file.name,
                )

                payload = {
                    "location": location.strip(),
                    "growth_stage": growth_stage,
                    "wine_type": wine_type.strip(),
                    "recent_treatments": recent_treatments,
                    "image": image_asset,
                }

                graph = deps["build_architecture"]()
                install_agent_progress_hooks(graph, deps["Node"], status_slot)

                raw_result = graph.invoke(payload)

            result = parse_graph_output(raw_result)
            predicted_disease = result.get("predicted_disease", "N/D")
            risk_level = result.get("risk_level", "N/D")
            decision_report = result.get("decision_report", "Nessun report disponibile.")

            st.success("Analisi completata.")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted disease", str(predicted_disease), border=True)
            with col_b:
                st.metric("Risk level", str(risk_level), border=True)

            safe_report = html.escape(str(decision_report)).replace("\n", "<br>")
            st.markdown("### Decision report")
            st.markdown(f'<div class="report-card">{safe_report}</div>', unsafe_allow_html=True)

        except Exception as exc:
            status_slot.error("Elaborazione interrotta.")
            st.error("Si e verificato un errore durante l'analisi. Riprova o controlla i dettagli tecnici.")
            with st.expander("Dettagli tecnici errore"):
                st.code(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()

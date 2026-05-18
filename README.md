# VERITAS

Sistema multi-agente per supporto decisionale in viticoltura.

Funzioni principali:
- classificazione malattia fogliare da immagine (CNN)
- contesto meteo locale
- retrieval semantico su linee guida/prodotti (Qdrant)
- sintesi decisionale finale con LLM (MASFactory)

## Architettura

Il grafo e definito in `architecture/masfactory_graph.py`.

Nodi principali:
1. `InputParserNode`
2. `VisionAgentNode`
3. `ContextAgentNode`
4. `ConditionalNode`
5. `RAGAgentNode` (solo se malattia != Healthy)
6. `BypassNode` (solo se malattia == Healthy)
7. `DecisionAgentNode`

## Requisiti

- Python 3.10+
- Windows + PowerShell
- accesso internet per API LLM e meteo

Installa dipendenze:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configurazione `.env`

Crea un file `.env` nella root del progetto con:

```env
OPENAI_API_KEY=your_key_here
BASE_URL=your_llm_base_url_here
```

## Cosa manca per poter runnare questa repo

Questi asset sono ignorati e quindi non arrivano da GitHub:
- `dataset/` (dataset completo immagini)
- `models/embeddings/` (modello embedding OpenVINO locale)
- `vector_store/` (database vettoriale Qdrant locale)
- `.env`

Asset invece presenti in repo:
- `models/modello_cnn_vine_disease-AdaptiveAvgPool2d--99.5.ckpt` (checkpoint CNN)
- `example_dataset/` (mini dataset per bootstrap rapido)
- chunk JSONL gia pronti in `data/guidelines/chunks/` e `data/products/chunks/`

## Bootstrap su macchina nuova (clone-and-run)

### 1) Scegli dataset per la CNN

Il codice usa `DATASET_DIR` in `config/settings.py` per leggere le classi.

Se non hai il dataset completo, usa subito `example_dataset`:
- imposta `DATASET_DIR = PROJECT_ROOT / "example_dataset"` in `config/settings.py`

Se hai il dataset completo, mantieni:
- `DATASET_DIR = PROJECT_ROOT / "dataset" / "Dataset-splittato" / "train"`
- dataset completo scaricabile da Kaggle: https://www.kaggle.com/datasets/leonardoserafinn/grapevine-desease-splitted

### 2) Prepara modello embedding locale (OpenVINO)

```powershell
py -3 scripts/export_embedding_openvino.py --verify
```

Questo popola `models/embeddings/qwen3-embedding-0.6b-openvino/`.

### 3) Ricrea il vector store locale Qdrant

```powershell
py -3 scripts/build_qdrant_index.py --target all --recreate
```

Questo popola `vector_store/qdrant_local/`.

### 4) Esegui end-to-end

```powershell
py -3 main.py
```

## Output runtime (monitor live)

Durante `main.py` vedrai log in tempo reale via hook MASFactory:
- `[START] <NodeName>`
- `[OUTPUT] <NodeName>: ...`
- `[END] <NodeName> (<ms>)`

Il risultato finale viene poi stampato con `pprint`.

## Avvia interfaccia Streamlit

```powershell
streamlit run app.py
```

## Script utili

Documentazione operativa completa in:
- `scripts/README.md`

Comandi piu usati:

```powershell
py -3 scripts/export_embedding_openvino.py --verify
py -3 scripts/build_qdrant_index.py --target all --recreate
py -3 scripts/test_rag_search.py --collection guidelines --query "peronospora vite fioritura" --top-k 5 --region Veneto
```

## Troubleshooting rapido

- Errore modello embedding locale non trovato:
  - riesegui `py -3 scripts/export_embedding_openvino.py --verify`
- Retrieval vuoto o inconsistente:
  - ricrea indice con `py -3 scripts/build_qdrant_index.py --target all --recreate`
- Errore su classi CNN/dataset mancante:
  - verifica `DATASET_DIR` in `config/settings.py`
  - per test rapido usa `example_dataset`
- Errore API meteo/LLM:
  - verifica connettivita e valori in `.env`

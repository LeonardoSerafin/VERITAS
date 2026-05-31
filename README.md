# VERITAS

VERITAS è un sistema multi-agente per supporto decisionale in viticoltura. Combina classificazione visiva, contesto ambientale, retrieval semantico su documenti agronomici/prodotti fitosanitari e sintesi finale tramite LLM.

Funzioni principali:
- preprocessing dell'immagine e validazione foglia di vite prima della diagnosi
- classificazione malattia fogliare da immagine con CNN dedicata
- normalizzazione del contesto agronomico e meteo
- retrieval RAG su linee guida e prodotti tramite Qdrant locale
- embedding Qwen3 con runtime automatico OpenVINO/PyTorch
- reranking Qwen3 dei candidati recuperati da Qdrant
- sintesi decisionale finale con MASFactory

## Architettura

Il grafo MASFactory è definito in `architecture/masfactory_graph.py`.

Nodi principali:
- `InputParserNode`: normalizza input utente.
- `VisionAgentNode`: pre-processa l'immagine, verifica che sia una foglia di vite analizzabile e poi classifica la malattia con CNN.
- `ContextAgentNode`: prepara contesto locale/meteo/agronomico.
- `ConditionalNode`: decide se attivare RAG o bypass in base a malattia e stato dell'immagine.
- `RAGAgentNode`: recupera evidenze documentali se l'immagine è valida e la malattia non è `Sana`.
- `BypassNode`: evita retrieval quando la foglia è sana o l'immagine non è utilizzabile.
- `DecisionAgentNode`: produce la risposta finale.

## Pipeline Visiva

Il `VisionAgentNode` gestisce l'intera pipeline visuale prima di produrre gli input per gli altri nodi:

1. `tools/image_preprocessing_tool.py` converte l'`ImageAsset` in `PIL.Image`, valuta qualità tecnica dell'immagine e applica correzioni leggere quando possibile.
2. Se il preprocessing giudica l'immagine non recuperabile, il VisionAgent restituisce `state = "unusable image"` e non chiama la CNN.
3. `tools/grape_leaf_validator_tool.py` usa un classificatore MobileNetV3 Small binario per distinguere `valid_grape_leaf` da `invalid_image`.
4. Se l'immagine non rappresenta una foglia di vite analizzabile, il VisionAgent restituisce `state = "invalid image"` e non chiama la CNN.
5. Solo con `state = "valid image"`, `tools/cnn_leaf_disease_tool.py` esegue la classificazione della malattia fogliare.

Stati principali prodotti dal VisionAgent:

- `valid image`: immagine valida, classificazione CNN eseguita.
- `invalid image`: immagine tecnicamente leggibile ma non riconosciuta come foglia di vite analizzabile.
- `unusable image`: immagine scartata dal preprocessing per qualita tecnica insufficiente.

Gli stati `invalid image` e `unusable image` vanno in bypass del RAG e il `DecisionAgentNode` chiede una nuova immagine senza produrre raccomandazioni agronomiche.

## Pipeline RAG

La pipeline RAG è composta da questi passaggi:

1. I documenti Markdown puliti vengono trasformati in chunk JSONL da `scripts/chunk_markdown_documents.py`.
2. I prodotti fitosanitari JSON vengono trasformati in chunk JSONL da `scripts/chunk_products_json.py`.
3. `scripts/build_qdrant_index.py` genera embedding dei chunk e popola Qdrant locale.
4. `tools/rag_retrieval_tool.py` recupera molti candidati da Qdrant usando embedding della query.
5. `tools/reranker_tool.py` riordina i candidati con Qwen3 Reranker.
6. Il RAGAgent riceve solo i migliori blocchi finali come contesto.

Separare retrieval e reranking è importante: Qdrant filtra velocemente molti candidati, il reranker decide quali sono davvero più rilevanti per la query.

## Requisiti

- Python 3.10+
- Windows + PowerShell, oppure ambiente compatibile con gli stessi comandi Python
- accesso internet per primo download modelli Hugging Face
- accesso alle API configurate in `.env`
- spazio disco sufficiente per modelli Qwen3 e Qdrant locale

Installa dipendenze base:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se vuoi usare OpenVINO su GPU Intel:

```powershell
pip install -r requirements-openvino.txt
```

## Configurazione `.env`

Crea un file `.env` nella root del progetto:

```env
OPENAI_API_KEY=your_key_here
BASE_URL=your_llm_base_url_here
```

## Asset Non Versionati

Questi asset sono ignorati da Git e vanno ricreati o copiati sulla macchina locale:
- `dataset/`: dataset completo immagini per CNN.
- `models/embeddings/`: modelli embedding esportati in OpenVINO.
- `models/rerankers/`: modelli reranker esportati in OpenVINO.
- `vector_store/`: database Qdrant locale.
- `.env`: chiavi e endpoint runtime.

Asset presenti in repo:
- `models/modello_cnn_vine_disease-AdaptiveAvgPool2d--99.5.ckpt`: checkpoint CNN.
- `models/grape_leaf_validator_mobilenetv3_small_best.pt`: checkpoint validator foglia di vite vs non foglia di vite.
- `example_dataset/`: mini dataset per bootstrap rapido.
- `data/guidelines/chunks/guideline_chunks.jsonl`: chunk linee guida già pronti.
- `data/products/chunks/product_chunks.jsonl`: chunk prodotti già pronti.

## Bootstrap Su Macchina Nuova

### 1. Configura Modelli Vision

I path e le classi vision sono configurati in `config/settings.py`.

Checkpoint CNN malattie:

```python
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "modello_cnn_vine_disease-AdaptiveAvgPool2d--99.5.ckpt"
DISEASE_CLASS_NAMES = [
    "Black Rot (Guignardia bidwelii)",
    "ESCA",
    "Sana",
    "Escoriosi (Phomopsis viticola)",
]
IMAGE_SIZE = 256
VISION_CNN_TOP_K = 4
```

Checkpoint validatore foglia di vite:

```python
VALIDATOR_MODEL_PATH = PROJECT_ROOT / "models" / "grape_leaf_validator_mobilenetv3_small_best.pt"
VALIDATOR_CLASS_TO_IDX = {
    "invalid_image": 0,
    "valid_grape_leaf": 1,
}
VALIDATOR_THRESHOLD = 0.85
VALIDATOR_DEVICE = "auto"
```

Dataset usato per addestrare il validatore MobileNetV3 Small: https://www.kaggle.com/datasets/leonardoserafinn/grapevine-leaf-vs-non-grapevine-leaf

La cartella `Non foglie di vite/` contiene 10 immagini negative di esempio estratte da quel dataset, utili per smoke test manuali del validator e del bypass del RAG.

`DATASET_DIR` resta configurabile per compatibilita con il tool CNN e per bootstrap locale.

Per test rapido senza dataset completo, imposta:

```python
DATASET_DIR = PROJECT_ROOT / "example_dataset"
```

Per uso completo, mantieni o ripristina:

```python
DATASET_DIR = PROJECT_ROOT / "dataset" / "Dataset-splittato" / "train"
```

Dataset completo: https://www.kaggle.com/datasets/leonardoserafinn/grapevine-desease-splitted

### 2. Scegli I Modelli RAG

I modelli si configurano in `config/settings.py`.

Configurazione leggera:

```python
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "embeddings" / "qwen3-embedding-0.6b-openvino"
EMBEDDING_VECTOR_SIZE = 1024
EMBEDDING_BATCH_SIZE = 4

RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "rerankers" / "qwen3-reranker-0.6b-openvino"
RERANKER_BATCH_SIZE = 4
```

Configurazione più pesante e qualitativa:

```python
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "embeddings" / "qwen3-embedding-4b-openvino"
EMBEDDING_VECTOR_SIZE = 2560
EMBEDDING_BATCH_SIZE = 1

RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
RERANKER_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "rerankers" / "qwen3-reranker-4b-openvino"
RERANKER_BATCH_SIZE = 1
```

Nota: quando cambi embedding model o `EMBEDDING_VECTOR_SIZE`, devi ricreare Qdrant da zero. Una collection costruita con vettori da `1024` non può ricevere vettori da `2560`.

### 3. Runtime Automatico Embedding E Reranker

Embedding e reranker usano la stessa logica runtime:
- `auto`: usa OpenVINO locale se trova GPU Intel e artifact esportati, altrimenti PyTorch.
- `openvino`: forza OpenVINO locale e fallisce se gli artifact mancano.
- `torch`: forza PyTorch e seleziona `cuda`, `mps` o `cpu`.

Config rilevanti:

```python
EMBEDDING_RUNTIME = "auto"
EMBEDDING_TORCH_DEVICE = "auto"
OPENVINO_DEVICE = "GPU"

RERANKER_RUNTIME = "auto"
RERANKER_TORCH_DEVICE = "auto"
RERANKER_OPENVINO_DEVICE = "GPU"
```

### 4. Esporta Modelli OpenVINO Locali

Questo step serve solo per GPU Intel/OpenVINO. Se non esporti i modelli, `auto` usa PyTorch.

Embedding:

```powershell
py -3 scripts/export_embedding_openvino.py --verify
```

Reranker:

```powershell
py -3 scripts/export_reranker_openvino.py --verify
```

Con modelli 4B puoi essere esplicito:

```powershell
py -3 scripts/export_embedding_openvino.py --model-id Qwen/Qwen3-Embedding-4B --output-dir models/embeddings/qwen3-embedding-4b-openvino --verify
py -3 scripts/export_reranker_openvino.py --model-id Qwen/Qwen3-Reranker-4B --output-dir models/rerankers/qwen3-reranker-4b-openvino --verify
```

### 5. Ricrea Qdrant Locale

Se `vector_store/qdrant_local/` non esiste o i modelli embedding non sono cambiati:

```powershell
py -3 scripts/build_qdrant_index.py --target all --recreate
```

Se hai cambiato dimensione embedding, per esempio da `1024` a `2560`, pulisci prima lo storage locale:

```powershell
Remove-Item -Recurse -Force ".\vector_store\qdrant_local"
py -3 scripts/build_qdrant_index.py --target all --recreate
```

Questo popola `vector_store/qdrant_local/` con collection compatibili con il modello embedding corrente.

### 6. Smoke Test Retrieval

```powershell
py -3 scripts/test_rag_search.py --collection guidelines --query "peronospora vite trattamenti rameici limitazioni uso" --top-k 8 --region Emilia-Romagna --document-type linee_guida
```

Per testare anche il reranker attraverso il retriever MASFactory, usa uno script temporaneo o una shell Python che istanzia `QdrantRetrieval` e controlla `block.metadata["vector_score"]` e `block.metadata["rerank_score"]`.

### 7. Esegui End-To-End

```powershell
py -3 main.py
```

Per interfaccia Streamlit:

```powershell
streamlit run app.py
```

## Output Runtime

Durante `main.py` vedrai log live via hook MASFactory:
- `[START] <NodeName>`
- `[OUTPUT] <NodeName>: ...`
- `[END] <NodeName> (<s>)`

Il risultato finale viene stampato con `pprint`.

## Script Utili

Documentazione operativa completa:
- `scripts/README.md`

Comandi frequenti:

```powershell
py -3 scripts/export_embedding_openvino.py --verify
py -3 scripts/export_reranker_openvino.py --verify
py -3 scripts/build_qdrant_index.py --target all --recreate
py -3 scripts/test_rag_search.py --collection guidelines --query "peronospora vite fioritura" --top-k 5 --region Veneto
```

## Troubleshooting Rapido

- Errore `could not broadcast input array from shape (2560,) into shape (1024,)`: hai cambiato modello embedding senza pulire Qdrant; esegui `Remove-Item -Recurse -Force ".\vector_store\qdrant_local"` e ricostruisci l'indice.
- OpenVINO non viene usato: verifica che esistano `models/embeddings/.../openvino/openvino_model.xml` o `models/rerankers/.../openvino/openvino_model.xml` e che `requirements-openvino.txt` sia installato.
- Primo run lento: Hugging Face sta scaricando modelli o OpenVINO sta compilando/cacheando il modello.
- Retrieval vuoto o inconsistente: ricrea indice con `py -3 scripts/build_qdrant_index.py --target all --recreate`.
- Errore su classi CNN/dataset mancante: verifica `DATASET_DIR` in `config/settings.py`, oppure usa `example_dataset`.
- Errore su checkpoint CNN: verifica `CNN_MODEL_PATH` in `config/settings.py` e la presenza del file in `models/`.
- Errore su validator foglia di vite: verifica `VALIDATOR_MODEL_PATH`, `VALIDATOR_CLASS_TO_IDX` e la presenza di `models/grape_leaf_validator_mobilenetv3_small_best.pt`.
- Immagine scartata prima della diagnosi: controlla `state` nel log del `VisionAgentNode`; `unusable image` indica qualità tecnica insufficiente, `invalid image` indica immagine non riconosciuta come foglia di vite analizzabile.
- Errore API meteo/LLM: verifica connettività, `OPENAI_API_KEY` e `BASE_URL` in `.env`.

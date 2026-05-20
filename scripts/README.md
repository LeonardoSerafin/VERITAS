
# Scripts Guide

Questo file documenta gli script in `scripts/`, con scopo, input, output, comandi consigliati e note operative. L'obiettivo è rendere riproducibile la pipeline RAG anche su una macchina nuova.

## Panoramica Veloce

Ordine tipico per ricostruire tutta la pipeline RAG:

1. `inline_html_tables.py`, solo per documenti OCR con tabelle HTML linkate.
2. `chunk_markdown_documents.py`, per generare chunk linee guida.
3. `chunk_products_json.py`, per generare chunk prodotti fitosanitari.
4. `export_embedding_openvino.py`, opzionale per usare embedding OpenVINO locale su GPU Intel.
5. `export_reranker_openvino.py`, opzionale per usare reranker OpenVINO locale su GPU Intel.
6. `build_qdrant_index.py`, per popolare Qdrant locale.
7. `test_rag_search.py`, per smoke test del retrieval vettoriale.

I file JSONL dei chunk sono già presenti in repo. Quindi su una macchina nuova spesso basta esportare i modelli OpenVINO, se servono, e ricostruire Qdrant.

## 1) export_embedding_openvino.py

### Scopo

Esporta una volta sola il modello embedding da Hugging Face in formato OpenVINO locale. VERITAS lo usa automaticamente quando `EMBEDDING_RUNTIME = "auto"`, viene rilevata una GPU Intel e gli artifact locali sono presenti.

### Input principali

- model id HF (`--model-id`, default da `config/settings.py`)
- output directory (`--output-dir`)
- device OpenVINO (`--device`)

### Output

Directory modello locale, tipicamente:

- `models/embeddings/qwen3-embedding-0.6b-openvino/`
- `models/embeddings/qwen3-embedding-4b-openvino/`

### Comandi

```powershell
pip install -r requirements-openvino.txt
py -3 scripts/export_embedding_openvino.py
py -3 scripts/export_embedding_openvino.py --verify
py -3 scripts/export_embedding_openvino.py --force
```

Per esportare esplicitamente il modello 4B:

```powershell
py -3 scripts/export_embedding_openvino.py --model-id Qwen/Qwen3-Embedding-4B --output-dir models/embeddings/qwen3-embedding-4b-openvino --verify
```

### Note

- Se il modello locale esiste già, senza `--force` non riesporta.
- `--verify` prova il caricamento locale subito dopo export.
- Senza modello locale OpenVINO, gli embedding usano automaticamente PyTorch con CUDA, MPS o CPU se `EMBEDDING_RUNTIME = "auto"`.
- Se cambi modello embedding, controlla sempre `EMBEDDING_VECTOR_SIZE` in `config/settings.py`.
- `Qwen/Qwen3-Embedding-0.6B` produce vettori da `1024` dimensioni.
- `Qwen/Qwen3-Embedding-4B` produce vettori da `2560` dimensioni.

## 2) export_reranker_openvino.py

### Scopo

Esporta una volta sola il modello reranker da Hugging Face in formato OpenVINO locale. VERITAS lo usa automaticamente quando `RERANKER_RUNTIME = "auto"`, viene rilevata una GPU Intel e gli artifact locali sono presenti.

Il reranker non genera embedding. Valuta coppie `(query, documento)` e assegna uno score di rilevanza ai candidati già recuperati da Qdrant.

### Input principali

- model id HF (`--model-id`, default da `config/settings.py`)
- output directory (`--output-dir`)
- device OpenVINO (`--device`)

### Output

Directory modello locale, tipicamente:

- `models/rerankers/qwen3-reranker-0.6b-openvino/`
- `models/rerankers/qwen3-reranker-4b-openvino/`

### Comandi

```powershell
pip install -r requirements-openvino.txt
py -3 scripts/export_reranker_openvino.py
py -3 scripts/export_reranker_openvino.py --verify
py -3 scripts/export_reranker_openvino.py --force
```

Per esportare esplicitamente il modello 4B:

```powershell
py -3 scripts/export_reranker_openvino.py --model-id Qwen/Qwen3-Reranker-4B --output-dir models/rerankers/qwen3-reranker-4b-openvino --verify
```

### Note

- Se il modello locale esiste già, senza `--force` non riesporta.
- `--verify` prova un caricamento locale e una prediction minima.
- Senza modello locale OpenVINO, il reranker usa PyTorch con CUDA, MPS o CPU se `RERANKER_RUNTIME = "auto"`.
- Il reranker 4B è più pesante: se hai problemi di memoria, riduci `RERANKER_BATCH_SIZE` in `config/settings.py`.

## 3) inline_html_tables.py

### Scopo

Sostituisce link a tabelle HTML nel Markdown OCR ottenuto tramite Mistral OCR con HTML inline reale (utile per preservare contenuto tabellare nei chunk).

### Input principali

- `--input-md`: markdown sorgente
- `--document-root`: root documento OCR (deve contenere `pages/`)
- `--output-md`: markdown risultante

### Output

Nuovo markdown autocontenuto, con blocchi:

- `TABLE_START ... TABLE_END`

### Comando esempio

```powershell
py -3 scripts/inline_html_tables.py `
  --input-md "data/guidelines/ocr_documents/<doc>/<doc>.md" `
  --document-root "data/guidelines/ocr_documents/<doc>" `
  --output-md "data/guidelines/clean_guidelines/<doc>.md"
```

### Note

- Se non trova una tabella, inserisce un marker `TABLE_NOT_FOUND`.

## 4) chunk_markdown_documents.py

### Scopo

Chunking semantico dei documenti markdown di linee guida, con:

- split per header
- preservazione blocchi tabellari HTML
- overlap configurato
- metadata arricchiti per retrieval

### Input principali (hardcoded nello script)

- `data/guidelines/clean_guidelines/*.md`
- `data/guidelines/document_metadata.json`

### Output

- `data/guidelines/chunks/guideline_chunks.jsonl`

### Comando

```powershell
py -3 scripts/chunk_markdown_documents.py
```

### Schema chunk (sintesi)

Top-level keys:

- `chunk_id`, `doc_id`, `source_file`, `title`, `text`, `metadata`

Metadata principali:

- `document_type`, `region`, `year`, `headers`, `has_table`, `table_sources`, `pages`, `keywords`, `chunk_index`, `chunk_char_length`

## 5) chunk_products_json.py

### Scopo

Trasforma l'elenco ministeriale prodotti fitosanitari (JSON raw) in chunk JSONL normalizzati per retrieval.

### Input principali

- `data/products/raw/elenco completo dei Prodotti Fitosanitari autorizzati dal Ministero della Salute italiano.json`

### Output

- `data/products/chunks/product_chunks.jsonl`

### Comando

```powershell
py -3 scripts/chunk_products_json.py
```

### Cosa fa

- normalizza campi testuali e date
- inferisce categorie (es. fungicide/insecticide)
- inferisce flag `possibly_vine_relevant`
- produce testo naturale ottimizzato per embedding

### Schema chunk (sintesi)

Top-level keys:

- `chunk_id`, `doc_id`, `source_file`, `text`, `metadata`, `raw_record`

Metadata principali:

- `product_name`, `registration_number`, `status`, `active_substances`, `categories`, `possibly_vine_relevant`, campi amministrativi/temporali

## 6) build_qdrant_index.py

### Scopo

Indicizza i chunk JSONL su Qdrant locale, generando vettori embedding con il runtime selezionato in `config/settings.py`.

Ordine di preferenza:

- GPU Intel + modello OpenVINO locale: OpenVINO locale
- GPU NVIDIA: PyTorch CUDA
- Apple Silicon: PyTorch MPS
- fallback: PyTorch CPU

### Input

- `guideline_chunks.jsonl`
- `product_chunks.jsonl`

### Output

- collection Qdrant locali in `vector_store/qdrant_local/`

### Comandi

```powershell
py -3 scripts/build_qdrant_index.py --target guidelines
py -3 scripts/build_qdrant_index.py --target products
py -3 scripts/build_qdrant_index.py --target all
py -3 scripts/build_qdrant_index.py --target all --recreate
```

### Opzioni

- `--target`: `guidelines | products | all`
- `--recreate`: cancella e ricrea la collection target prima dell'upsert

### Cambio Modello Embedding

Se cambi modello embedding o `EMBEDDING_VECTOR_SIZE`, devi ricostruire Qdrant in modo pulito. Esempio: passando da `Qwen3-Embedding-0.6B` a `Qwen3-Embedding-4B`, la dimensione vettoriale passa da `1024` a `2560`.

Pulizia completa consigliata:

```powershell
Remove-Item -Recurse -Force ".\vector_store\qdrant_local"
py -3 scripts/build_qdrant_index.py --target all --recreate
```

Errore tipico se non pulisci Qdrant:

```text
ValueError: could not broadcast input array from shape (2560,) into shape (1024,)
```

Questo significa che Qdrant locale contiene ancora strutture create per il vecchio modello embedding.

## 7) test_rag_search.py

### Scopo

Smoke test retrieval vettoriale su collection Qdrant con query testuale. Questo script testa Qdrant + embedding query. Il reranking MASFactory è invece applicato da `tools/rag_retrieval_tool.py` quando viene usato `QdrantRetrieval`.

### Comandi esempio

```powershell
py -3 scripts/test_rag_search.py --collection guidelines --query "peronospora vite fioritura" --top-k 5 --region Veneto
py -3 scripts/test_rag_search.py --collection products --query "fungicida rame" --top-k 5 --status authorized
```

### Filtri disponibili

- `--status`
- `--region`
- `--document-type`

## Troubleshooting script-level

- Vuoi usare OpenVINO su GPU Intel: installa `pip install -r requirements-openvino.txt`, poi esegui `py -3 scripts/export_embedding_openvino.py --verify` e `py -3 scripts/export_reranker_openvino.py --verify`.
- OpenVINO non viene selezionato in `auto`: verifica che esista `openvino_model.xml` sotto la directory configurata in `EMBEDDING_MODEL_LOCAL_PATH` o `RERANKER_MODEL_LOCAL_PATH`.
- Errore dimensione vettore in indicizzazione: allinea `EMBEDDING_VECTOR_SIZE` con il modello reale e cancella `vector_store/qdrant_local` prima di ricostruire.
- Retrieval vuoto: verifica presenza chunk e collection, poi prova `py -3 scripts/build_qdrant_index.py --target all --recreate`.
- Primo run molto lento: Hugging Face sta scaricando modelli o OpenVINO sta compilando/cacheando gli artifact.
- Modelli 4B troppo lenti o out-of-memory: riduci `EMBEDDING_BATCH_SIZE`, `RERANKER_BATCH_SIZE` e `RAG_CANDIDATE_TOP_K_GUIDELINES`.

## Config RAG Di Riferimento

Le impostazioni principali sono in `config/settings.py`.

Embedding 0.6B:

```python
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "embeddings" / "qwen3-embedding-0.6b-openvino"
EMBEDDING_VECTOR_SIZE = 1024
EMBEDDING_BATCH_SIZE = 4
```

Embedding 4B:

```python
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "embeddings" / "qwen3-embedding-4b-openvino"
EMBEDDING_VECTOR_SIZE = 2560
EMBEDDING_BATCH_SIZE = 1
```

Reranker 0.6B:

```python
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "rerankers" / "qwen3-reranker-0.6b-openvino"
RERANKER_BATCH_SIZE = 4
```

Reranker 4B:

```python
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
RERANKER_MODEL_LOCAL_PATH = PROJECT_ROOT / "models" / "rerankers" / "qwen3-reranker-4b-openvino"
RERANKER_BATCH_SIZE = 1
```

Retrieval + reranking:

```python
RAG_RERANK_ENABLED = True
RAG_TOP_K_GUIDELINES = 8
RAG_CANDIDATE_TOP_K_GUIDELINES = 50
```

`RAG_CANDIDATE_TOP_K_GUIDELINES` indica quanti candidati recupera Qdrant prima del reranking. `RAG_TOP_K_GUIDELINES` indica quanti blocchi finali arrivano all'agente.

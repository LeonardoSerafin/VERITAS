
# Scripts Guide

Questo file documenta tutti gli script presenti in `scripts/`, con obiettivo, input, output e comandi consigliati.

## Panoramica veloce

Ordine tipico di utilizzo:

1. `chunk_markdown_documents.py`
2. `chunk_products_json.py`
3. `build_qdrant_index.py`
4. `test_rag_search.py`

`inline_html_tables.py` e uno script di pre-processing opzionale prima del chunking dei Markdown OCR.
`export_embedding_openvino.py` e opzionale: serve solo per preparare l'accelerazione OpenVINO su GPU Intel.

## 1) export_embedding_openvino.py

### Scopo

Esporta una volta sola il modello embedding da Hugging Face in formato OpenVINO locale.
VERITAS lo usera automaticamente quando rileva una GPU Intel e trova gli artifact locali.

### Input principali

- model id HF (`--model-id`, default da `config/settings.py`)
- output directory (`--output-dir`)
- device OpenVINO (`--device`)

### Output

Directory modello locale, tipicamente:

- `models/embeddings/qwen3-embedding-0.6b-openvino/`

### Comandi

```powershell
pip install -r requirements-openvino.txt
py -3 scripts/export_embedding_openvino.py
py -3 scripts/export_embedding_openvino.py --verify
py -3 scripts/export_embedding_openvino.py --force
```

### Note

- Se il modello locale esiste gia, senza `--force` non riesporta.
- `--verify` prova il caricamento locale subito dopo export.
- Senza modello locale OpenVINO, gli embedding usano automaticamente PyTorch con CUDA, MPS o CPU.

## 2) inline_html_tables.py

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

## 3) chunk_markdown_documents.py

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

## 4) chunk_products_json.py

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

## 5) build_qdrant_index.py

### Scopo

Indicizza i chunk JSONL su Qdrant locale, generando vettori embedding con il runtime selezionato automaticamente.

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

## 6) test_rag_search.py

### Scopo

Smoke test retrieval su collection Qdrant con query testuale.

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

- Vuoi usare OpenVINO su GPU Intel:
  - installare `pip install -r requirements-openvino.txt`
  - rieseguire `py -3 scripts/export_embedding_openvino.py --verify`
- Errore dimensione vettore in indicizzazione:
  - allineare `EMBEDDING_VECTOR_SIZE` con modello reale
- Retrieval vuoto:
  - verificare presenza chunk e collection
  - provare `--recreate` su `build_qdrant_index.py`

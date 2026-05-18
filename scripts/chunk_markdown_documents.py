from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "guidelines" / "clean_guidelines"
OUTPUT_PATH = PROJECT_ROOT / "data" / "guidelines" / "chunks" /  "guideline_chunks.jsonl"
METADATA_PATH = PROJECT_ROOT / "data" / "guidelines" / "document_metadata.json"

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 500

TABLE_BLOCK_RE = re.compile(
    r"<!--\s*TABLE_START(?P<meta>.*?)-->(?P<body>.*?)<!--\s*TABLE_END\s*-->",
    re.DOTALL | re.IGNORECASE,
)

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
PAGE_RE = re.compile(r"page=[\"']?(?P<page>\d+)[\"']?", re.IGNORECASE)
SOURCE_RE = re.compile(r"source=[\"'](?P<source>[^\"']+)[\"']", re.IGNORECASE)


def load_document_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Metadata file non trovato: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[\s_]+", "_", value)
    return value.strip("_")


def extract_title(markdown: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback


def extract_table_metadata(text: str) -> Dict[str, Any]:
    table_sources = []
    pages = set()

    for match in TABLE_BLOCK_RE.finditer(text):
        raw_meta = match.group("meta") or ""

        source_match = SOURCE_RE.search(raw_meta)
        if source_match:
            table_sources.append(source_match.group("source"))

        page_match = PAGE_RE.search(raw_meta)
        if page_match:
            pages.add(int(page_match.group("page")))

    return {
        "has_table": bool(table_sources) or "<table" in text.lower(),
        "table_sources": table_sources,
        "pages": sorted(pages),
    }


def infer_keywords(text: str) -> Dict[str, bool]:
    lowered = text.lower()

    keywords = {
        "peronospora": "peronospora" in lowered,
        "bornitura": "bornitura" in lowered,
        "black rot": "black rot" in lowered or "blackrot" in lowered,
        "mal dell'esca": "mal dell'esca" in lowered,
        "leaf blight": "leaf blight" in lowered or "leafblight" in lowered,
        "vite": "vite" in lowered or "vitis" in lowered,
        "intervallo_sicurezza": "intervallo di sicurezza" in lowered,
        "dose": "dose" in lowered or "dosaggio" in lowered,
        "trattamento": "trattamento" in lowered or "trattamenti" in lowered or "treatment" in lowered,
    }

    return keywords


def split_markdown_by_headers(markdown: str) -> List[Dict[str, Any]]:
    """
    Divide il Markdown in sezioni preservando la gerarchia degli header.

    Output:
    [
        {
            "headers": {"h1": "...", "h2": "..."},
            "text": "..."
        }
    ]
    """
    matches = list(HEADER_RE.finditer(markdown))

    if not matches:
        return [{"headers": {}, "text": markdown.strip()}]

    sections = []
    current_headers: Dict[str, str] = {}

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)

        section_text = markdown[start:end].strip()

        # Aggiorna gerarchia header.
        current_headers[f"h{level}"] = title

        # Rimuove header più profondi quando si torna a un livello superiore.
        for deeper_level in range(level + 1, 7):
            current_headers.pop(f"h{deeper_level}", None)

        if section_text:
            sections.append(
                {
                    "headers": dict(current_headers),
                    "text": section_text,
                }
            )

    return sections


def split_into_blocks_preserving_tables(text: str) -> List[str]:
    """
    Spezza il testo in blocchi, cercando di non separare le tabelle HTML marcate
    con TABLE_START / TABLE_END.
    """
    blocks: List[str] = []
    position = 0

    for match in TABLE_BLOCK_RE.finditer(text):
        before = text[position:match.start()]
        table = match.group(0)

        blocks.extend(split_plain_text_into_paragraphs(before))
        blocks.append(table.strip())

        position = match.end()

    after = text[position:]
    blocks.extend(split_plain_text_into_paragraphs(after))

    return [block for block in blocks if block.strip()]


def split_plain_text_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def hard_split_long_block(block: str, max_size: int) -> List[str]:
    """
    Se un blocco è più grande del chunk size, lo spezza brutalmente.
    Serve come fallback per tabelle enormi o sezioni OCR molto lunghe.
    """
    if len(block) <= max_size:
        return [block]

    pieces = []
    start = 0

    while start < len(block):
        end = start + max_size
        pieces.append(block[start:end].strip())
        start = end

    return pieces


def build_chunk_text(headers: Dict[str, str], body: str) -> str:
    header_lines = []

    for level in range(1, 7):
        key = f"h{level}"
        if key in headers:
            header_lines.append(f"{'#' * level} {headers[key]}")

    if header_lines:
        return "\n".join(header_lines) + "\n\n" + body.strip()

    return body.strip()


def chunk_section(
    section_text: str,
    headers: Dict[str, str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    blocks = split_into_blocks_preserving_tables(section_text)

    chunks = []
    current = ""

    for block in blocks:
        candidate_blocks = hard_split_long_block(block, chunk_size)

        for candidate in candidate_blocks:
            if not current:
                current = candidate
                continue

            proposed = current + "\n\n" + candidate

            if len(proposed) <= chunk_size:
                current = proposed
            else:
                chunks.append(build_chunk_text(headers, current))

                overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                current = (overlap_text + "\n\n" + candidate).strip()

    if current.strip():
        chunks.append(build_chunk_text(headers, current))

    return chunks


def make_chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}_{index:05d}"


def process_markdown_file(
    md_path: Path,
    doc_metadata: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    raw_text = md_path.read_text(encoding="utf-8")
    markdown = normalize_text(raw_text)

    fallback_doc_id = slugify(md_path.stem)
    doc_id = doc_metadata.get("doc_id", fallback_doc_id)
    title = extract_title(markdown, fallback=md_path.stem)

    sections = split_markdown_by_headers(markdown)

    chunks: List[Dict[str, Any]] = []
    local_index = 0

    for section in sections:
        headers = section["headers"]
        section_text = section["text"]

        section_chunks = chunk_section(
            section_text=section_text,
            headers=headers,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_text in section_chunks:
            table_meta = extract_table_metadata(chunk_text)
            keyword_meta = infer_keywords(chunk_text)

            chunk = {
                "chunk_id": make_chunk_id(doc_id, local_index),
                "doc_id": doc_id,
                "source_file": md_path.name,
                "title": title,
                "text": chunk_text,
                "metadata": {
                    **doc_metadata,
                    "doc_id": doc_id,
                    "source_file": md_path.name,
                    "title": title,
                    "headers": headers,
                    "chunk_index": local_index,
                    "chunk_char_length": len(chunk_text),
                    "has_table": table_meta["has_table"],
                    "table_sources": table_meta["table_sources"],
                    "pages": table_meta["pages"],
                    "keywords": keyword_meta,
                },
            }

            chunks.append(chunk)
            local_index += 1

    return chunks


def save_jsonl(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def print_stats(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        print("Nessun chunk generato.")
        return

    lengths = [len(chunk["text"]) for chunk in chunks]
    table_chunks = [chunk for chunk in chunks if chunk["metadata"].get("has_table")]

    print("\n=== CHUNKING STATS ===")
    print(f"Chunk totali: {len(chunks)}")
    print(f"Lunghezza media: {sum(lengths) / len(lengths):.1f} caratteri")
    print(f"Lunghezza minima: {min(lengths)} caratteri")
    print(f"Lunghezza massima: {max(lengths)} caratteri")
    print(f"Chunk con tabelle: {len(table_chunks)}")
    print("======================\n")


def main() -> None:
    metadata_by_file = load_document_metadata(METADATA_PATH)

    md_files = sorted(INPUT_DIR.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"Nessun file .md trovato in {INPUT_DIR}")

    all_chunks: List[Dict[str, Any]] = []

    for md_path in md_files:
        print(f"[INFO] Processing {md_path.name}")

        doc_metadata = metadata_by_file.get(md_path.name, {})

        if not doc_metadata:
            print(
                f"[WARN] Nessun metadata manuale trovato per {md_path.name}. "
                "Uso metadata automatici minimi."
            )

        chunks = process_markdown_file(
            md_path=md_path,
            doc_metadata=doc_metadata,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        print(f"[INFO] Generati {len(chunks)} chunk da {md_path.name}")
        all_chunks.extend(chunks)

    save_jsonl(all_chunks, OUTPUT_PATH)
    print_stats(all_chunks)

    print(f"[OK] Salvato: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
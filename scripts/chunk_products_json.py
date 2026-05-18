from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_JSON_PATH = PROJECT_ROOT / "data" / "products" / "raw" / "elenco completo dei Prodotti Fitosanitari autorizzati dal Ministero della Salute italiano.json"
OUTPUT_JSONL_PATH = PROJECT_ROOT / "data" / "products" / "chunks" / "product_chunks.jsonl"


def normalize_string(value: Any) -> Optional[str]:
    """
    Normalizza stringhe vuote, '-' e None.
    """
    if value is None:
        return None

    value = str(value).strip()

    if not value or value == "-":
        return None

    return value


def normalize_upper(value: Any) -> Optional[str]:
    value = normalize_string(value)
    if value is None:
        return None
    return value.upper()


def parse_date_it(value: Any) -> Optional[str]:
    """
    Converte date italiane dd/mm/yyyy in yyyy-mm-dd.
    Se non riesce, restituisce None.
    """
    value = normalize_string(value)

    if value is None:
        return None

    match = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", value)

    if not match:
        return None

    day, month, year = match.groups()
    return f"{year}-{month}-{day}"


def split_active_substances(value: Any) -> List[str]:
    """
    Divide sostanze attive scritte in un campo unico.

    Esempi possibili:
    - "THIOPHANATE-METHYL"
    - "RAME METALLO, ZOLFO"
    - "FOSETIL-AL + MANCOZEB"
    """
    value = normalize_string(value)

    if value is None:
        return []

    parts = re.split(r"\s*(?:,|;|\+|/|\be\b|\bE\b)\s*", value)

    cleaned = []
    for part in parts:
        part = part.strip().upper()
        if part and part not in {"-", "N.A.", "NA"}:
            cleaned.append(part)

    # Rimuove duplicati preservando ordine.
    seen = set()
    unique = []
    for item in cleaned:
        if item not in seen:
            unique.append(item)
            seen.add(item)

    return unique


def normalize_status(value: Any) -> str:
    """
    Normalizza lo stato amministrativo in valori più comodi per filtri.
    """
    value_norm = normalize_upper(value)

    if value_norm is None:
        return "unknown"

    if "AUTORIZZATO" in value_norm:
        return "authorized"

    if "REVOCATO" in value_norm:
        return "revoked"

    if "SOSPESO" in value_norm:
        return "suspended"

    if "SCADUTO" in value_norm:
        return "expired"

    return value_norm.lower().replace(" ", "_")


def parse_yes_no(value: Any) -> Optional[bool]:
    value_norm = normalize_upper(value)

    if value_norm is None:
        return None

    if value_norm == "SI" or value_norm == "SÌ" or value_norm == "YES":
        return True

    if value_norm == "NO":
        return False

    return None


def infer_product_categories(record: Dict[str, Any]) -> List[str]:
    """
    Euristica semplice su formulazione, attività e sostanze.
    Non è una classificazione ufficiale.
    Serve solo per retrieval/filtering leggero.
    """
    text = " ".join(
        str(record.get(field, "") or "")
        for field in [
            "denominazione_prodotto",
            "attivita",
            "descrizione_formulazione",
            "sostanze_attive",
            "indicazioni_di_pericolo",
        ]
    ).lower()

    categories = []

    rules = {
        "fungicide": [
            "fungicida",
            "antiperonosporico",
            "antioidico",
            "rame",
            "copper",  
            "zolfo",
            "sulphur",  
            "mancozeb",
            "fosetil",
            "metalaxyl",
            "cymoxanil",
            "dimetomorf",
            "folpet",
            "tebuconazolo",
            "difenoconazolo",
            "dicloran",  
        ],
        "insecticide": [
            "insetticida",
            "diazinon",  
            "aficida",
            "larvicida",
            "acetamiprid",
            "deltametrina",
            "lambda",
            "spinosad",
            "abamectina",
        ],
        "acaricide": [
            "acaricida",
            "dicofol",  
            "dinocap",  
            "abamectina",
            "exitiazox",
            "tebufenpirad",
        ],
        "adjuvant": [  
            "coadiuvante",
            "agente bagnante",
        ],
    }

    for category, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            categories.append(category)

    return categories


def infer_vine_relevance(record: Dict[str, Any]) -> bool:
    """
    Euristica molto leggera: utile solo come segnale, non come verità normativa.
    Attenzione: l'assenza di 'vite' nel record prodotto non vuol dire che non sia autorizzato su vite.
    """
    text = " ".join(
        str(record.get(field, "") or "")
        for field in [
            "denominazione_prodotto",
            "attivita",
            "descrizione_formulazione",
            "sostanze_attive",
        ]
    ).lower()

    vine_related_keywords = [
        "vite",
        "uva",
        "vigneto",
        "peronospora",
        "black rot",
        "mal dell'esca",
        "oidio",
        "botrite",
        "zolfo",
        "rame",
        "folpet",
        "fosetil",
        "metalaxyl",
        "cymoxanil",
        "dimetomorf",
    ]

    return any(keyword in text for keyword in vine_related_keywords)


def build_product_text(record: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """
    Testo naturale da embeddare.
    Deve essere leggibile dal retriever e dall'LLM.
    """
    lines = []

    product_name = metadata.get("product_name")
    registration_number = metadata.get("registration_number")
    company = metadata.get("company")
    status = metadata.get("status_raw")
    active_substances = metadata.get("active_substances", [])

    lines.append(f"Prodotto fitosanitario: {product_name or 'Nome non disponibile'}.")

    if registration_number:
        lines.append(f"Numero di registrazione: {registration_number}.")

    if company:
        lines.append(f"Titolare o ragione sociale: {company}.")

    if status:
        lines.append(f"Stato amministrativo: {status}.")

    if active_substances:
        lines.append(
            "Sostanze attive: " + ", ".join(active_substances) + "."
        )

    formulation_code = metadata.get("formulation_code")
    formulation_description = metadata.get("formulation_description")

    if formulation_code or formulation_description:
        lines.append(
            f"Formulazione: {formulation_code or ''} "
            f"{formulation_description or ''}.".strip()
        )

    hazard = metadata.get("hazard_indications")
    if hazard:
        lines.append(f"Indicazioni di pericolo: {hazard}.")

    activity = metadata.get("activity")
    if activity:
        lines.append(f"Attività dichiarata: {activity}.")

    registration_date = metadata.get("registration_date")
    expiration_date = metadata.get("authorization_expiration_date")

    if registration_date:
        lines.append(f"Data registrazione: {registration_date}.")

    if expiration_date:
        lines.append(f"Data scadenza autorizzazione: {expiration_date}.")

    revocation_reason = metadata.get("revocation_reason")
    revocation_decree_date = metadata.get("revocation_decree_date")
    revocation_effective_date = metadata.get("revocation_effective_date")

    if revocation_reason:
        lines.append(f"Motivo della revoca: {revocation_reason}.")

    if revocation_decree_date:
        lines.append(f"Data decreto revoca: {revocation_decree_date}.")

    if revocation_effective_date:
        lines.append(f"Data decorrenza revoca: {revocation_effective_date}.")

    if metadata.get("is_parallel_import") is not None:
        lines.append(
            f"Importazione parallela: "
            f"{'sì' if metadata['is_parallel_import'] else 'no'}."
        )

    if metadata.get("pfnpo") is not None:
        lines.append(f"PFnPO: {'sì' if metadata['pfnpo'] else 'no'}.")

    if metadata.get("pfnpe") is not None:
        lines.append(f"PFnPE: {'sì' if metadata['pfnpe'] else 'no'}.")

    categories = metadata.get("categories", [])
    if categories:
        lines.append("Categorie inferite: " + ", ".join(categories) + ".")

    lines.append(
        "Nota: questo record descrive il prodotto registrato, non necessariamente "
        "i singoli impieghi autorizzati per coltura, avversità, dose o intervallo di sicurezza."
    )

    return "\n".join(lines)


def build_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    registration_number = normalize_string(record.get("num_registrazione"))
    product_name = normalize_string(record.get("denominazione_prodotto"))

    status_raw = normalize_string(record.get("stato_amministrativo"))
    status = normalize_status(status_raw)

    active_substances = split_active_substances(record.get("sostanze_attive"))

    metadata = {
        "chunk_type": "plant_protection_product",
        "registration_number": registration_number,
        "product_name": product_name,
        "company": normalize_string(record.get("ragione_sociale")),

        "legal_address": {
            "address": normalize_string(record.get("indirizzo_sede_legale")),
            "zip": normalize_string(record.get("cap_sede_legale")),
            "city": normalize_string(record.get("comune_sede_legale")),
            "province": normalize_string(record.get("provincia_sede_legale")),
        },

        "administrative_address": {
            "address": normalize_string(record.get("indirizzo_sede_amministrativa")),
            "zip": normalize_string(record.get("cap_sede_amministrativa")),
            "city": normalize_string(record.get("comune_sede_amministrativa")),
            "province": normalize_string(record.get("provincia_sede_amministrativa")),
        },

        "registration_date": parse_date_it(record.get("data_registrazione")),
        "authorization_expiration_date": parse_date_it(
            record.get("data_scadenza_autorizzazione")
        ),

        "hazard_indications": normalize_string(record.get("indicazioni_di_pericolo")),
        "activity": normalize_string(record.get("attivita")),

        "formulation_code": normalize_string(record.get("codice_formulazione")),
        "formulation_description": normalize_string(
            record.get("descrizione_formulazione")
        ),

        "active_substances": active_substances,
        "active_substances_raw": normalize_string(record.get("sostanze_attive")),
        "content_per_100g": normalize_string(
            record.get("contenuto_per_100g_di_prodotto")
        ),

        "is_parallel_import": parse_yes_no(record.get("importazione_parallela")),
        "pfnpo": parse_yes_no(record.get("PFnPO")),
        "pfnpe": parse_yes_no(record.get("PFnPE")),

        "status": status,
        "status_raw": status_raw,

        "revocation_reason": normalize_string(
            record.get("motivo_della revoca")
            or record.get("motivo_della_revoca")
        ),
        "revocation_decree_date": parse_date_it(record.get("data_decreto_revoca")),
        "revocation_effective_date": parse_date_it(
            record.get("data_decorrenza_revoca")
        ),
    }

    metadata["categories"] = infer_product_categories(record)
    metadata["possibly_vine_relevant"] = infer_vine_relevance(record)

    return metadata


def make_chunk_id(metadata: Dict[str, Any], index: int) -> str:
    registration_number = metadata.get("registration_number")

    if registration_number:
        safe_reg = re.sub(r"[^\w-]", "_", registration_number)
        return f"product_{safe_reg}"

    return f"product_{index:06d}"


def chunk_product_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    metadata = build_metadata(record)
    text = build_product_text(record, metadata)

    chunk_id = make_chunk_id(metadata, index)

    return {
        "chunk_id": chunk_id,
        "doc_id": "prodotti_fitosanitari_ministero",
        "source_file": INPUT_JSON_PATH.name,
        "text": text,
        "metadata": {
            **metadata,
            "source_file": INPUT_JSON_PATH.name,
            "record_index": index,
            "chunk_char_length": len(text),
        },
        "raw_record": record,
    }


def load_products_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Mi aspettavo un JSON con struttura lista: [{...}, {...}, ...]"
        )

    return data


def save_jsonl(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def print_stats(chunks: List[Dict[str, Any]]) -> None:
    total = len(chunks)

    if total == 0:
        print("Nessun chunk generato.")
        return

    status_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    active_substance_counts: Dict[str, int] = {}

    vine_relevant = 0

    for chunk in chunks:
        metadata = chunk["metadata"]

        status = metadata.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        if metadata.get("possibly_vine_relevant"):
            vine_relevant += 1

        for category in metadata.get("categories", []):
            category_counts[category] = category_counts.get(category, 0) + 1

        for substance in metadata.get("active_substances", []):
            active_substance_counts[substance] = (
                active_substance_counts.get(substance, 0) + 1
            )

    top_substances = sorted(
        active_substance_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    print("\n=== PRODUCT CHUNKING STATS ===")
    print(f"Chunk totali: {total}")
    print(f"Possibilmente rilevanti per vite: {vine_relevant}")

    print("\nStati amministrativi:")
    for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {status}: {count}")

    print("\nCategorie inferite:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {category}: {count}")

    print("\nTop 20 sostanze attive:")
    for substance, count in top_substances:
        print(f"  - {substance}: {count}")

    print("===============================\n")


def main() -> None:
    print(f"[INFO] Lettura JSON: {INPUT_JSON_PATH}")
    records = load_products_json(INPUT_JSON_PATH)

    print(f"[INFO] Record letti: {len(records)}")

    chunks = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            print(f"[WARN] Record {index} non è un oggetto JSON. Saltato.")
            continue

        chunk = chunk_product_record(record, index)
        chunks.append(chunk)

    save_jsonl(chunks, OUTPUT_JSONL_PATH)
    print_stats(chunks)

    print(f"[OK] Salvato: {OUTPUT_JSONL_PATH}")


if __name__ == "__main__":
    main()
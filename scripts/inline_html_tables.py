import argparse
import re
from pathlib import Path
from typing import Optional


TABLE_LINK_PATTERN = re.compile(
    r"\[([^\]]+?\.html)\]\(([^)]+?\.html)\)"
)


def detect_current_page(line: str, current_page: Optional[int]) -> Optional[int]:
    """
    Prova a capire la pagina corrente leggendo il Markdown.

    Supporta pattern tipo:
    - # Page 12
    - ## Page 12
    - # Pagina 12
    - <!-- page: 12 -->
    - page-12

    Se il tuo Markdown ha un formato diverso, aggiungi un pattern qui.
    """

    patterns = [
        r"<!--\s*page\s*:\s*(\d+)\s*-->",
        r"#+\s*Page\s+(\d+)",
        r"#+\s*Pagina\s+(\d+)",
        r"\bpage-(\d+)\b",
        r"\bPagina\s+(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return current_page


def find_html_table_file(
    table_filename: str,
    document_root: Path,
    current_page: Optional[int] = None,
) -> Optional[Path]:
    """
    Cerca il file HTML della tabella.

    Prima prova nella pagina corrente:
        pages/page-12/tbl-506.html

    Poi, se non lo trova, cerca ricorsivamente dentro document_root.
    """

    candidate_paths = []

    if current_page is not None:
        candidate_paths.extend(
            [
                document_root / "pages" / f"page-{current_page}" / table_filename,
                document_root / "pages" / f"page-{current_page:03d}" / table_filename,
                document_root / "pages" / str(current_page) / table_filename,
            ]
        )

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    matches = list(document_root.rglob(table_filename))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(
            f"[WARNING] Trovati più file chiamati {table_filename}. "
            f"Uso il primo: {matches[0]}"
        )
        return matches[0]

    return None


def inline_html_tables(
    markdown_text: str,
    document_root: Path,
    wrap_in_markers: bool = True,
) -> str:
    """
    Sostituisce i link alle tabelle HTML con il contenuto HTML effettivo.
    """

    output_lines = []
    current_page: Optional[int] = None

    missing_tables = []

    for line in markdown_text.splitlines():
        current_page = detect_current_page(line, current_page)

        def replace_match(match: re.Match) -> str:
            label = match.group(1)
            href = match.group(2)

            table_filename = Path(href).name

            html_path = find_html_table_file(
                table_filename=table_filename,
                document_root=document_root,
                current_page=current_page,
            )

            if html_path is None:
                missing_tables.append(
                    {
                        "table_filename": table_filename,
                        "page": current_page,
                        "original_reference": match.group(0),
                    }
                )

                return (
                    f"\n\n<!-- TABLE_NOT_FOUND "
                    f"filename=\"{table_filename}\" "
                    f"page=\"{current_page}\" -->\n"
                    f"{match.group(0)}\n"
                )

            html_content = html_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            try:
                relative_path = html_path.relative_to(document_root)
            except ValueError:
                relative_path = html_path

            if wrap_in_markers:
                return (
                    "\n\n"
                    f"<!-- TABLE_START source=\"{relative_path}\" page=\"{current_page}\" -->\n\n"
                    f"{html_content}\n\n"
                    "<!-- TABLE_END -->\n\n"
                )

            return f"\n\n{html_content}\n\n"

        new_line = TABLE_LINK_PATTERN.sub(replace_match, line)
        output_lines.append(new_line)

    if missing_tables:
        print("\n[WARNING] Tabelle non trovate:")
        for item in missing_tables:
            print(
                f"- {item['table_filename']} "
                f"(page={item['page']}) "
                f"reference={item['original_reference']}"
            )

    return "\n".join(output_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sostituisce i link a tabelle HTML in un Markdown con il contenuto HTML reale."
    )

    parser.add_argument(
        "--input-md",
        required=True,
        help="Path del file Markdown grande generato da OCR.",
    )

    parser.add_argument(
        "--document-root",
        required=True,
        help=(
            "Cartella root del documento OCR. "
            "Deve contenere il Markdown e la directory pages/."
        ),
    )

    parser.add_argument(
        "--output-md",
        required=True,
        help="Path del nuovo Markdown autocontenuto da generare.",
    )

    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Non aggiunge commenti TABLE_START / TABLE_END attorno all'HTML.",
    )

    args = parser.parse_args()

    input_md = Path(args.input_md)
    document_root = Path(args.document_root)
    output_md = Path(args.output_md)

    if not input_md.exists():
        raise FileNotFoundError(f"File Markdown non trovato: {input_md}")

    if not document_root.exists():
        raise FileNotFoundError(f"Document root non trovata: {document_root}")

    markdown_text = input_md.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    processed_markdown = inline_html_tables(
        markdown_text=markdown_text,
        document_root=document_root,
        wrap_in_markers=not args.no_markers,
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_md.write_text(
        processed_markdown,
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Input Markdown: {input_md}")
    print(f"Document root: {document_root}")
    print(f"Output Markdown: {output_md}")


if __name__ == "__main__":
    main()
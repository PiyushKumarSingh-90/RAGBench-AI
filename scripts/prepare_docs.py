"""Documentation preparation script for RAGBench Phase 1.

Reads locally collected documentation from data/raw/ across all five sources:
- python
- scikit_learn
- docker
- fastapi
- aws

Processes raw documents into a unified, structured JSON Lines file:
data/processed/documents.jsonl

Generates the summary corpus manifest:
data/processed/corpus_manifest.json

Standard library only. No external downloads or internet access.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class SimpleHTMLToMarkdown(HTMLParser):
    """HTML to clean Markdown converter using standard library html.parser."""

    def __init__(self, main_container_id: str | None = None) -> None:
        super().__init__()
        self.main_container_id = main_container_id
        self.in_target_container = main_container_id is None
        self.container_depth = 0

        self.title: str = ""
        self.in_title = False

        self.headings: list[dict[str, Any]] = []
        self.current_heading_level: int | None = None
        self.current_heading_text: list[str] = []

        self.chunks: list[str] = []
        self.in_pre = False
        self.in_code = False
        self.in_table = False
        self.table_rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: list[str] = []
        self.is_header_cell = False
        self.has_table_header = False

        self.list_stack: list[dict[str, Any]] = []
        self.in_link: str | None = None
        self.link_text: list[str] = []

        self.ignore_tags = {"script", "style", "noscript", "svg", "button", "footer", "nav"}
        self.in_ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        if self.main_container_id and not self.in_target_container:
            tag_id = attr_dict.get("id", "")
            if tag_id == self.main_container_id:
                self.in_target_container = True
                self.container_depth = 1
                return

        if self.main_container_id and self.in_target_container:
            self.container_depth += 1

        if not self.in_target_container:
            if tag == "title":
                self.in_title = True
            return

        if tag in self.ignore_tags:
            self.in_ignored += 1
            return
        if self.in_ignored > 0:
            return

        if tag == "title":
            self.in_title = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.current_heading_level = int(tag[1])
            self.current_heading_text = []
            self.chunks.append("\n\n" + "#" * self.current_heading_level + " ")
        elif tag == "p":
            self.chunks.append("\n\n")
        elif tag == "pre":
            self.in_pre = True
            self.chunks.append("\n\n```\n")
        elif tag == "code" and not self.in_pre:
            self.in_code = True
            self.chunks.append("`")
        elif tag == "ul":
            self.list_stack.append({"type": "ul", "count": 0})
            self.chunks.append("\n")
        elif tag == "ol":
            self.list_stack.append({"type": "ol", "count": 1})
            self.chunks.append("\n")
        elif tag == "li":
            self.chunks.append("\n")
            if self.list_stack:
                indent = "  " * (len(self.list_stack) - 1)
                parent = self.list_stack[-1]
                if parent["type"] == "ol":
                    self.chunks.append(f"{indent}{parent['count']}. ")
                    parent["count"] += 1
                else:
                    self.chunks.append(f"{indent}- ")
        elif tag == "a":
            href = attr_dict.get("href")
            if href and not href.startswith("#"):
                self.in_link = href
                self.link_text = []
        elif tag == "table":
            self.in_table = True
            self.table_rows = []
            self.chunks.append("\n\n")
        elif tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.is_header_cell = tag == "th"
            self.current_cell = []
        elif tag == "br":
            self.chunks.append("\n")
        elif tag == "blockquote":
            self.chunks.append("\n\n> ")

    def handle_endtag(self, tag: str) -> None:
        if self.main_container_id and self.in_target_container:
            self.container_depth -= 1
            if self.container_depth <= 0:
                self.in_target_container = False
                return

        if not self.in_target_container:
            if tag == "title":
                self.in_title = False
            return

        if tag in self.ignore_tags:
            if self.in_ignored > 0:
                self.in_ignored -= 1
            return
        if self.in_ignored > 0:
            return

        if tag == "title":
            self.in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = "".join(self.current_heading_text).strip()
            level = self.current_heading_level or 1
            if text:
                self.headings.append({"level": level, "text": text})
                if not self.title and level == 1:
                    self.title = text
            self.current_heading_level = None
            self.current_heading_text = []
            self.chunks.append("\n\n")
        elif tag == "pre":
            self.in_pre = False
            self.chunks.append("\n```\n\n")
        elif tag == "code" and not self.in_pre:
            self.in_code = False
            self.chunks.append("`")
        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self.chunks.append("\n")
        elif tag == "a":
            if self.in_link:
                text = "".join(self.link_text).strip() or "link"
                self.chunks.append(f"[{text}]({self.in_link})")
                self.in_link = None
                self.link_text = []
        elif tag in ("td", "th"):
            cell_text = "".join(self.current_cell).strip().replace("\n", " ")
            self.current_row.append(cell_text)
            if self.is_header_cell:
                self.has_table_header = True
        elif tag == "tr":
            if self.current_row:
                self.table_rows.append(self.current_row)
        elif tag == "table":
            self.in_table = False
            if self.table_rows:
                # Render markdown table
                col_count = max(len(r) for r in self.table_rows)
                normalized_rows = [r + [""] * (col_count - len(r)) for r in self.table_rows]
                header = normalized_rows[0]
                self.chunks.append("| " + " | ".join(header) + " |\n")
                self.chunks.append("| " + " | ".join(["---"] * col_count) + " |\n")
                for row in normalized_rows[1:]:
                    self.chunks.append("| " + " | ".join(row) + " |\n")
                self.chunks.append("\n")
            self.table_rows = []

    def handle_data(self, data: str) -> None:
        if self.in_title and not self.title:
            self.title = data.strip().split(" - ")[0].split(" | ")[0]

        if not self.in_target_container or self.in_ignored > 0:
            return

        if self.current_heading_level is not None:
            self.current_heading_text.append(data)
            self.chunks.append(data)
            return

        if self.in_link:
            self.link_text.append(data)
            return

        if self.in_table and hasattr(self, "current_cell"):
            self.current_cell.append(data)
            return

        if self.in_pre:
            self.chunks.append(data)
        else:
            self.chunks.append(data)

    def get_markdown(self) -> str:
        raw = "".join(self.chunks)
        # Clean up excessive blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", raw)
        return cleaned.strip()


def parse_html_document(
    raw_html: str, main_container_id: str | None = None
) -> tuple[str, list[dict[str, Any]], str]:
    """Parse HTML and extract title, ordered headings, and clean markdown content."""
    parser = SimpleHTMLToMarkdown(main_container_id=main_container_id)
    parser.feed(raw_html)
    parser.close()
    content = parser.get_markdown()
    title = parser.title or "Untitled Document"
    headings = parser.headings
    if not headings and title:
        headings = [{"level": 1, "text": title}]
    return title, headings, content


def parse_markdown_document(raw_md: str) -> tuple[str, list[dict[str, Any]], str]:
    """Parse Markdown and extract title, ordered headings, and normalized content."""
    lines = raw_md.splitlines()
    title = ""
    headings: list[dict[str, Any]] = []

    # Strip frontmatter if present (e.g. Hugo / Jekyll)
    in_frontmatter = False
    frontmatter_title = ""
    content_lines: list[str] = []

    for i, line in enumerate(lines):
        if i == 0 and line.strip() == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if line.strip() == "---":
                in_frontmatter = False
                continue
            if line.startswith("title:"):
                frontmatter_title = line.split("title:", 1)[1].strip().strip("\"'")
            continue
        content_lines.append(line)

    # Process content lines for headings and clean formatting
    in_code_block = False
    cleaned_content_lines: list[str] = []
    for line in content_lines:
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_code_block = not in_code_block
            cleaned_content_lines.append(line)
            continue
        if not in_code_block:
            # Strip Hugo shortcodes like {{< youtube-embed ... >}}
            line = re.sub(r"\{\{<[^>]+>\}\}", "", line)
            trimmed = line.strip()
            if trimmed.startswith("#"):
                match = re.match(r"^(#{1,6})\s+(.+)$", trimmed)
                if match:
                    level = len(match.group(1))
                    text = match.group(2).strip()
                    # Clean links and MkDocs attribute anchors {#custom-anchor}
                    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
                    text = re.sub(r"\s*\{[#\.\s][^\}]*\}", "", text).strip()
                    headings.append({"level": level, "text": text})
                    if not title and level == 1:
                        title = text
                    line = f"{'#' * level} {text}"
        cleaned_content_lines.append(line)

    if not title:
        title = frontmatter_title or (headings[0]["text"] if headings else "Untitled Document")

    # If document has frontmatter title but no top-level # heading, add it to headings
    if headings and headings[0]["text"] != title and frontmatter_title:
        headings.insert(0, {"level": 1, "text": title})
    elif not headings and title:
        headings.append({"level": 1, "text": title})

    content = "\n".join(cleaned_content_lines).strip()
    return title, headings, content


def parse_rst_document(raw_rst: str) -> tuple[str, list[dict[str, Any]], str]:
    """Parse reStructuredText (.rst) document into title, headings, and clean Markdown."""
    lines = raw_rst.splitlines()
    headings: list[dict[str, Any]] = []
    title = ""

    # RST heading markers in order of typical level
    underlines = ["=", "-", "~", "^", '"', "#", "*"]
    md_lines: list[str] = []

    level_map: dict[str, int] = {}
    next_level = 1

    in_code = False
    i = 0
    total = len(lines)

    while i < total:
        line = lines[i]
        stripped = line.strip()

        # Check for overline + underline (title style)
        if (
            i + 2 < total
            and len(stripped) >= 3
            and all(c == stripped[0] for c in stripped)
            and stripped[0] in underlines
        ):
            marker = stripped[0]
            candidate_title = lines[i + 1].strip()
            closing = lines[i + 2].strip()
            if closing == stripped and candidate_title:
                if marker not in level_map:
                    level_map[marker] = next_level
                    next_level += 1
                level = level_map[marker]
                headings.append({"level": level, "text": candidate_title})
                if not title:
                    title = candidate_title
                md_lines.append(f"\n{'#' * level} {candidate_title}\n")
                i += 3
                continue

        # Check for underline only (standard RST heading)
        if (
            i + 1 < total
            and stripped
            and not stripped.startswith("..")
            and not stripped.startswith("::")
        ):
            next_line = lines[i + 1].strip()
            if (
                len(next_line) >= len(stripped)
                and len(next_line) >= 3
                and all(c == next_line[0] for c in next_line)
                and next_line[0] in underlines
            ):
                marker = next_line[0]
                if marker not in level_map:
                    level_map[marker] = next_level
                    next_level += 1
                level = level_map[marker]
                heading_text = stripped
                # Strip inline roles from heading text
                heading_text = re.sub(r":[a-zA-Z0-9_-]+:`([^`]+)`", r"\1", heading_text)
                headings.append({"level": level, "text": heading_text})
                if not title:
                    title = heading_text
                md_lines.append(f"\n{'#' * level} {heading_text}\n")
                i += 2
                continue

        # Convert code-block directive
        if stripped.startswith(".. code-block::"):
            lang = stripped.split("::", 1)[1].strip() or ""
            md_lines.append(f"\n```{lang}")
            in_code = True
            i += 1
            continue
        elif stripped.endswith("::") and not stripped.startswith(".."):
            prefix = stripped[:-2].rstrip()
            if prefix:
                md_lines.append(prefix)
            md_lines.append("\n```")
            in_code = True
            i += 1
            continue

        # Clean standard RST admonitions: .. note::, .. warning::, .. tip::
        admonition_match = re.match(
            r"^\.\.\s+(note|warning|tip|important|caution)::\s*(.*)$", stripped, re.IGNORECASE
        )
        if admonition_match:
            adm_type = admonition_match.group(1).capitalize()
            rest = admonition_match.group(2)
            md_lines.append(f"\n> **{adm_type}:** {rest}")
            i += 1
            continue

        # Skip comment or metadata directives
        if stripped.startswith(".. ") and not in_code:
            # Skip sphinx indexes, labels, comments
            i += 1
            continue

        # Transform inline RST roles: :func:`name` -> `name`, ``code`` -> `code`
        transformed = line
        transformed = re.sub(r":[a-zA-Z0-9_-]+:`([^`]+)`", r"`\1`", transformed)
        transformed = re.sub(r"``([^`]+)``", r"`\1`", transformed)
        # Transform RST links: `Title <URL>`_ -> [Title](URL)
        transformed = re.sub(r"`([^`<]+)<([^>]+)>`_", r"[\1](\2)", transformed)

        md_lines.append(transformed)
        i += 1

    content = "\n".join(md_lines)
    # Clean excessive blank lines
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if not title:
        title = headings[0]["text"] if headings else "Untitled Document"
    if not headings and title:
        headings.append({"level": 1, "text": title})

    return title, headings, content


def generate_doc_id(source_name: str, rel_path: str) -> str:
    """Generate a clean, stable document ID."""
    stem = Path(rel_path).stem
    # replace unwanted symbols with underscore
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    # include parent folder if it provides key context (e.g. tutorial, concepts)
    parent = Path(rel_path).parent.name
    if parent and parent not in ("raw", source_name, "docs", "tutorial", "content", "doc"):
        parent_slug = re.sub(r"[^a-zA-Z0-9]+", "_", parent).strip("_").lower()
        return f"{source_name}_{parent_slug}_{slug}"
    return f"{source_name}_{slug}"


def process_raw_file(
    file_path: Path,
    source_name: str,
    raw_root: Path,
    source_url_map: dict[str, str],
) -> dict[str, Any] | None:
    """Process a single raw documentation file into the required JSON schema."""
    rel_path = file_path.relative_to(raw_root).as_posix()

    # Ignore license files, gitkeep, readme in raw
    if file_path.name in (
        ".gitkeep",
        "LICENSE",
        "LICENSE.txt",
        "README.md",
        "source_manifest.json",
    ):
        return None

    suffix = file_path.suffix.lower()
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")

    if suffix in (".md", ".markdown"):
        title, headings, content = parse_markdown_document(raw_text)
    elif suffix == ".rst":
        title, headings, content = parse_rst_document(raw_text)
    elif suffix in (".html", ".htm"):
        main_id = "main-col-body" if source_name == "aws" else None
        title, headings, content = parse_html_document(raw_text, main_container_id=main_id)
    else:
        return None

    # Normalization: ensure unix line endings and clean whitespace
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_content:
        return None

    # Calculate deterministic hash and word count
    content_sha256 = hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()
    words = [w for w in re.split(r"\s+", normalized_content) if w]
    word_count = len(words)

    # Document ID and source URL
    doc_id = generate_doc_id(source_name, rel_path)
    source_url = source_url_map.get(rel_path) or f"https://example.com/{source_name}/{rel_path}"

    return {
        "document_id": doc_id,
        "source_name": source_name,
        "title": title,
        "source_path": f"{source_name}/{file_path.name}",
        "source_url": source_url,
        "language": "en",
        "headings": headings,
        "content_markdown": normalized_content,
        "word_count": word_count,
        "content_sha256": content_sha256,
    }


def main() -> None:
    """Main execution: process raw docs and generate JSONL + corpus manifest."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = raw_dir / "source_manifest.json"
    url_map: dict[str, str] = {}
    if source_manifest_path.exists():
        try:
            sm_data = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            for src in sm_data.get("sources", []):
                for page in src.get("included_pages", []):
                    rel = page.get("relative_path")
                    url = page.get("url")
                    if rel and url:
                        url_map[rel] = url
        except Exception as e:
            print(f"Warning: could not parse source_manifest.json: {e}")

    sources = ["python", "scikit_learn", "docker", "fastapi", "aws"]
    all_documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    excluded_files: list[dict[str, str]] = []
    docs_per_source: dict[str, int] = {s: 0 for s in sources}
    words_per_source: dict[str, int] = {s: 0 for s in sources}

    raw_files_count = 0

    for source_name in sources:
        src_raw_path = raw_dir / source_name
        if not src_raw_path.exists():
            continue

        # Sort files for deterministic processing
        files = sorted(list(src_raw_path.rglob("*")))
        for f in files:
            if f.is_file():
                if f.name in (
                    ".gitkeep",
                    "LICENSE",
                    "LICENSE.txt",
                    "README.md",
                    "source_manifest.json",
                ):
                    continue
                raw_files_count += 1
                doc = process_raw_file(f, source_name, raw_dir, url_map)
                if doc is None:
                    excluded_files.append(
                        {"file": f.as_posix(), "reason": "Unsupported format or empty content"}
                    )
                    continue

                doc_id = doc["document_id"]
                if doc_id in seen_ids:
                    duplicates.append(doc_id)
                    # Disambiguate if needed
                    doc["document_id"] = f"{doc_id}_{len(all_documents)}"
                seen_ids.add(doc["document_id"])

                all_documents.append(doc)
                docs_per_source[source_name] += 1
                words_per_source[source_name] += doc["word_count"]

    # Write data/processed/documents.jsonl
    jsonl_path = processed_dir / "documents.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for doc in all_documents:
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    dataset_size_bytes = jsonl_path.stat().st_size if jsonl_path.exists() else 0
    total_words = sum(doc["word_count"] for doc in all_documents)

    # Validate output
    errors: list[str] = []
    if len(all_documents) == 0:
        errors.append("No documents processed.")
    for s in sources:
        if docs_per_source[s] == 0:
            errors.append(f"Source '{s}' has 0 processed documents.")

    corpus_manifest = {
        "project": "RAGBench",
        "processing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_count": len(sources),
        "raw_files_discovered": raw_files_count,
        "processed_document_count": len(all_documents),
        "documents_per_source": docs_per_source,
        "words_per_source": words_per_source,
        "total_word_count": total_words,
        "dataset_size_bytes": dataset_size_bytes,
        "duplicate_count": len(duplicates),
        "excluded_files": excluded_files,
        "preparation_script": "scripts/prepare_docs.py (v1.0)",
        "source_manifest_reference": "data/raw/source_manifest.json",
        "validation": {
            "passed": len(errors) == 0,
            "errors": errors,
        },
    }

    corpus_manifest_path = processed_dir / "corpus_manifest.json"
    corpus_manifest_path.write_text(
        json.dumps(corpus_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Successfully processed {len(all_documents)} documents from {len(sources)} sources.")
    print(f"Total words: {total_words}")
    print(f"JSONL path: {jsonl_path} ({dataset_size_bytes} bytes)")
    print(f"Corpus manifest path: {corpus_manifest_path}")
    if errors:
        print(f"Validation warnings/errors: {errors}")


if __name__ == "__main__":
    main()

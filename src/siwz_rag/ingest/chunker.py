"""HTML-aware chunker dla publikacji Cortex (output cortex-docs-sync).

To jest centralna zmiana v4 vs v3. v3 traktował dokumentację jak płaski tekst:
  1. PDF/DOCX → Docling → markdown
  2. RecursiveCharacterTextSplitter → chunks 2200 chars
  3. Embeddings → Qdrant

Problemy które v3 miał:
  - Tabele kompatybilności OS były dzielone w połowie, tracąc nagłówki kolumn
    (= retrieval nie wiedział do której wersji agenta przypisana jest dana komórka).
  - Brak `heading_path` — chunk "Windows 10 21H2 — ✓ 9.2" nie wiedział, że jest
    w kontekście "Cortex XDR Agent Compatibility > Windows Desktop".
  - Brak deep-linków do portalu — źródło to była nazwa pliku PDF.

v4 chunker:
  - Parsuje HTML wygenerowany przez `cortex_docs_sync.html_assembly`.
  - Każda `<section>` = jeden topic publikacji (z h2 + breadcrumb + topic_url).
  - W obrębie sekcji: h3 = subsection (push do heading_path).
  - <table> = ATOMOWY chunk — nigdy nie dzielony (nawet jeśli > target_chars,
    aż do hard_max_chars; powyżej tego dzielimy po row-groups zachowując header).
  - Pozostałe bloki (p, ul, ol, pre) są mergowane do bieżącego chunka aż osiągniemy
    target_chars; wtedy flushujemy.
  - heading_path jest opcjonalnie wstrzykiwany jako prefix do tekstu chunka
    (boost dla embeddera — semantyka sekcji jest zakodowana w wektorze).

Metadane per chunk (zapisywane w payload Qdrant):
  - chunk_id (deterministyczny: hash z map_id + topic_index + block_index)
  - map_id (cortex-docs-sync publication id)
  - publication_title
  - product (XDR/XSIAM/XSOAR/XPANSE)
  - source_url (deep-link do publikacji w portalu)
  - topic_url (deep-link do topica)
  - topic_title, breadcrumb
  - heading_path (lista: [topic_title, h3_title?, h4_title?])
  - block_type ("text" | "table" | "list")
  - text (faktyczna treść, opcjonalnie z prefixem heading_path)
  - char_count, table_rows (gdy applicable)
  - source_file (lokalna ścieżka HTML — fallback gdy URL nie działa)
  - last_edition, diff_key (z manifestu cortex-docs-sync)
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag


# ── Klasa wynikowa ─────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """Pojedynczy chunk wyciągnięty z jednej publikacji HTML."""

    chunk_id: str
    map_id: str
    publication_title: str
    product: str
    source_url: str
    topic_url: str
    topic_title: str
    breadcrumb: str
    heading_path: List[str]
    block_type: str  # text | table | list
    text: str
    char_count: int
    source_file: str
    last_edition: str = ""
    diff_key: str = ""
    table_rows: int = 0


@dataclass
class PublicationMeta:
    """Wyciągnięte z nagłówka HTML metadane publikacji."""

    map_id: str
    title: str
    source_url: str
    products: List[str] = field(default_factory=list)
    category: str = ""
    version: str = ""
    last_edition: str = ""


# ── Konfiguracja chunkera ──────────────────────────────────────────────────


@dataclass
class ChunkerConfig:
    target_chars: int = 1800
    min_chars: int = 200
    hard_max_chars: int = 7000
    inject_heading_path: bool = True


# ── Funkcje pomocnicze ─────────────────────────────────────────────────────


_FT_BREADCRUMB_LABEL = re.compile(r"^\s*Breadcrumb\s*:\s*", re.IGNORECASE)
_FT_TOPIC_URL_LABEL = re.compile(r"^\s*Topic\s+URL\s*:\s*", re.IGNORECASE)


def _normalize_ws(text: str) -> str:
    """Skrócenie ciągów whitespace + normalizacja unicode."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _table_to_markdown(table: Tag) -> tuple[str, int]:
    """Konwersja <table> → markdown. Zwraca (text, n_rows)."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [_normalize_ws(td.get_text(separator=" ")) for td in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return "", 0

    # Wyrównanie szerokości kolumn (każdy wiersz ma >= max_cols)
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    md_lines = ["| " + " | ".join(r) + " |" for r in rows]
    # Separator po pierwszym wierszu (zakładamy że to header — tak generuje cortex-docs-sync)
    md_lines.insert(1, "| " + " | ".join(["---"] * max_cols) + " |")
    return "\n".join(md_lines), len(rows)


def _list_to_lines(node: Tag) -> str:
    """Konwersja <ul>/<ol> do bullet-listy tekstowej (zachowuje strukturę)."""
    is_ordered = node.name == "ol"
    out: list[str] = []
    for i, li in enumerate(node.find_all("li", recursive=False), 1):
        text = _normalize_ws(li.get_text(separator=" "))
        if not text:
            continue
        prefix = f"{i}. " if is_ordered else "- "
        out.append(prefix + text)
    return "\n".join(out)


def _block_text(node: Tag) -> str:
    """Wyciągnij tekst z dowolnego bloku, zachowując semantykę list i akapitów."""
    if node.name in ("ul", "ol"):
        return _list_to_lines(node)
    if node.name == "pre":
        # Bloki kodu — zachowaj jako fenced
        return "```\n" + node.get_text() + "\n```"
    return _normalize_ws(node.get_text(separator=" "))


def _hash_id(*parts: str) -> str:
    """Deterministyczny ID chunka — UUID-shape z SHA1 (Qdrant akceptuje UUID stringi)."""
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:32]
    # Formatuj jako UUID dla zgodności z Qdrant (akceptuje uuid albo int)
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _slug_product_dir(path: Path) -> str:
    """Zwróć identyfikator produktu z drugiego segmentu ścieżki: data/cortex_docs/xdr/foo.html → XDR."""
    parts = path.parts
    # Szukamy najbliżej pliku katalogu xdr|xsiam|xsoar|xpanse
    for seg in reversed(parts):
        s = seg.lower()
        if s in ("xdr", "xsiam", "xsoar", "xpanse"):
            return s.upper()
    return "UNKNOWN"


# ── Główny chunker ─────────────────────────────────────────────────────────


class HtmlChunker:
    """Chunker HTML-aware dla publikacji Cortex (cortex-docs-sync).

    Użycie:
        cfg = ChunkerConfig()
        chunker = HtmlChunker(cfg)
        for chunk in chunker.chunk_file(Path("data/cortex_docs/xdr/Cortex-XDR-Documentation__abc.html")):
            print(chunk.chunk_id, chunk.text[:80])
    """

    def __init__(self, cfg: ChunkerConfig | None = None) -> None:
        self.cfg = cfg or ChunkerConfig()

    # ── Public API ──────────────────────────────────────────────────────────

    def chunk_file(
        self,
        path: Path,
        diff_key: str = "",
    ) -> List[Chunk]:
        """Wczytaj plik HTML i zwróć listę chunków."""
        html = path.read_text(encoding="utf-8")
        return self.chunk_html(html, source_file=str(path), diff_key=diff_key)

    def chunk_html(
        self,
        html: str,
        source_file: str = "",
        diff_key: str = "",
    ) -> List[Chunk]:
        """Sparsuj HTML i wygeneruj listę chunków."""
        soup = BeautifulSoup(html, "lxml")
        meta = self._extract_publication_meta(soup)
        product = _slug_product_dir(Path(source_file)) if source_file else "UNKNOWN"

        chunks: List[Chunk] = []
        sections = soup.find_all("section")
        for topic_idx, section in enumerate(sections):
            section_chunks = self._chunk_section(
                section=section,
                topic_idx=topic_idx,
                meta=meta,
                product=product,
                source_file=source_file,
                diff_key=diff_key,
            )
            chunks.extend(section_chunks)

        return chunks

    # ── Wyciąganie metadanych publikacji ────────────────────────────────────

    def _extract_publication_meta(self, soup: BeautifulSoup) -> PublicationMeta:
        title = ""
        source_url = ""
        products: list[str] = []
        category = ""
        version = ""
        last_edition = ""
        map_id = ""

        h1 = soup.find("h1")
        if h1:
            title = _normalize_ws(h1.get_text())

        # Top-level <p> z metadanymi — przed <hr>
        for p in soup.find_all("p"):
            strong = p.find("strong")
            if not strong:
                continue
            label = strong.get_text(strip=True).rstrip(":").lower()
            if label.startswith("source"):
                a = p.find("a")
                if a and a.get("href"):
                    source_url = a["href"]
                    # map_id wyciągamy z URL: ostatni segment po /r/ ścieżki to zwykle pretty_url
                    # Ale stabilniejsze: weź z filename (cortex-docs-sync dodaje __<map_id>.html)
            elif label.startswith("product"):
                p_text = p.get_text(separator=" ", strip=True)
                # "Product(s): Cortex XDR, Cortex XDR Agent"
                if ":" in p_text:
                    rhs = p_text.split(":", 1)[1].strip()
                    products = [s.strip() for s in rhs.split(",") if s.strip()]
            elif label.startswith("category"):
                category = p.get_text(separator=" ", strip=True).split(":", 1)[-1].strip()
            elif label.startswith("version"):
                version = p.get_text(separator=" ", strip=True).split(":", 1)[-1].strip()
            elif label.startswith("last edition"):
                last_edition = p.get_text(separator=" ", strip=True).split(":", 1)[-1].strip()

        # map_id z URL: cortex-docs-sync filename = "<title>__<map_id>.html"
        # Najbezpieczniej: extract z source_url path tail jeśli możliwe, inaczej fallback
        if not map_id and source_url:
            # source_url typu: https://docs-cortex.../r/Cortex-XSIAM/Cortex-XSIAM-Documentation
            # nie zawiera map_id wprost. map_id wyciągniemy z filename w chunk_file().
            pass

        return PublicationMeta(
            map_id=map_id,
            title=title,
            source_url=source_url,
            products=products,
            category=category,
            version=version,
            last_edition=last_edition,
        )

    # ── Chunking pojedynczej sekcji (topica) ────────────────────────────────

    def _chunk_section(
        self,
        section: Tag,
        topic_idx: int,
        meta: PublicationMeta,
        product: str,
        source_file: str,
        diff_key: str,
    ) -> List[Chunk]:
        # 1. Topic metadata
        topic_title = ""
        breadcrumb = ""
        topic_url = ""

        h2 = section.find("h2")
        if h2:
            topic_title = _normalize_ws(h2.get_text())

        # Iteruj po PIERWSZYCH <p> w sekcji, szukając metadata Topic URL / Breadcrumb
        meta_paragraphs: set[int] = set()
        for idx, p in enumerate(section.find_all("p", recursive=False)):
            text = p.get_text(separator=" ", strip=True)
            if _FT_BREADCRUMB_LABEL.match(text):
                breadcrumb = _FT_BREADCRUMB_LABEL.sub("", text).strip()
                meta_paragraphs.add(id(p))
            elif _FT_TOPIC_URL_LABEL.match(text):
                a = p.find("a")
                if a and a.get("href"):
                    topic_url = a["href"]
                else:
                    topic_url = _FT_TOPIC_URL_LABEL.sub("", text).strip()
                meta_paragraphs.add(id(p))

        # 2. Iteruj po dzieciach sekcji w kolejności DOM
        heading_path: List[str] = [topic_title] if topic_title else []
        accumulated: list[str] = []
        accumulated_size = 0
        block_index = 0
        chunks: List[Chunk] = []

        # map_id z filename (cortex-docs-sync gwarantuje że jest tam __<map_id>.html)
        derived_map_id = ""
        if source_file:
            stem = Path(source_file).stem
            if "__" in stem:
                derived_map_id = stem.rsplit("__", 1)[-1]

        def _make_chunk(
            text: str,
            block_type: str,
            heading_path_snapshot: List[str],
            table_rows: int = 0,
        ) -> Chunk:
            nonlocal block_index
            block_index += 1
            payload_text = text
            if self.cfg.inject_heading_path and heading_path_snapshot:
                prefix = " > ".join(heading_path_snapshot)
                # Nie duplikuj jeśli tekst już zaczyna się od prefiksu (np. tabele już mają)
                if not payload_text.startswith(prefix):
                    payload_text = f"{prefix}\n\n{payload_text}"
            return Chunk(
                chunk_id=_hash_id(derived_map_id, str(topic_idx), str(block_index)),
                map_id=derived_map_id,
                publication_title=meta.title,
                product=product,
                source_url=meta.source_url,
                topic_url=topic_url,
                topic_title=topic_title,
                breadcrumb=breadcrumb,
                heading_path=list(heading_path_snapshot),
                block_type=block_type,
                text=payload_text,
                char_count=len(payload_text),
                source_file=source_file,
                last_edition=meta.last_edition,
                diff_key=diff_key,
                table_rows=table_rows,
            )

        def _flush_text() -> None:
            nonlocal accumulated, accumulated_size
            if not accumulated:
                return
            text = "\n\n".join(accumulated).strip()
            if len(text) < self.cfg.min_chars and chunks and chunks[-1].block_type == "text":
                # Merge z poprzednim chunkiem (uniknij mikro-fragmentów)
                prev = chunks[-1]
                merged = (prev.text + "\n\n" + text).strip()
                if len(merged) <= self.cfg.hard_max_chars:
                    chunks[-1] = Chunk(
                        chunk_id=prev.chunk_id,
                        map_id=prev.map_id,
                        publication_title=prev.publication_title,
                        product=prev.product,
                        source_url=prev.source_url,
                        topic_url=prev.topic_url,
                        topic_title=prev.topic_title,
                        breadcrumb=prev.breadcrumb,
                        heading_path=prev.heading_path,
                        block_type="text",
                        text=merged,
                        char_count=len(merged),
                        source_file=prev.source_file,
                        last_edition=prev.last_edition,
                        diff_key=prev.diff_key,
                    )
                    accumulated = []
                    accumulated_size = 0
                    return
            if text:
                chunks.append(_make_chunk(text, "text", list(heading_path)))
            accumulated = []
            accumulated_size = 0

        def _add_block_to_accumulated(text: str) -> None:
            """Dodaj blok do bufora; flush gdy bufor + nowy blok > target."""
            nonlocal accumulated, accumulated_size
            if not text:
                return
            if accumulated_size + len(text) > self.cfg.target_chars and accumulated:
                _flush_text()
            accumulated.append(text)
            accumulated_size += len(text) + 2  # \n\n separator

        for child in section.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue
            if id(child) in meta_paragraphs:
                continue
            if child.name == "h2":
                # Topic title już w heading_path
                continue
            if child.name == "h3":
                # Subheading — flush dotychczasowy buffer i zresetuj heading_path
                _flush_text()
                heading_path = [topic_title] if topic_title else []
                h3_text = _normalize_ws(child.get_text())
                if h3_text:
                    heading_path.append(h3_text)
                continue
            if child.name == "h4":
                # Sub-sub-heading — flush + push do path
                _flush_text()
                # Trzymaj h2 i h3 z path, dodaj h4
                h4_text = _normalize_ws(child.get_text())
                if h4_text:
                    if len(heading_path) >= 3:
                        heading_path = heading_path[:2] + [h4_text]
                    else:
                        heading_path = heading_path + [h4_text]
                continue
            if child.name == "table":
                # ATOMOWY: domknij text-buffer, dodaj tabelę jako oddzielny chunk
                _flush_text()
                table_md, n_rows = _table_to_markdown(child)
                if not table_md:
                    continue
                # Jeśli tabela > hard_max → podziel po row-groups z zachowanym headerem
                if len(table_md) > self.cfg.hard_max_chars and n_rows > 4:
                    for part in self._split_large_table(table_md, self.cfg.hard_max_chars):
                        chunks.append(_make_chunk(part, "table", list(heading_path), n_rows))
                else:
                    chunks.append(_make_chunk(table_md, "table", list(heading_path), n_rows))
                continue
            if child.name in ("ul", "ol"):
                # Listy — zachowujemy strukturę, ale traktujemy jako text-block
                list_text = _list_to_lines(child)
                _add_block_to_accumulated(list_text)
                continue
            if child.name in ("p", "div", "blockquote"):
                text = _block_text(child)
                _add_block_to_accumulated(text)
                continue
            if child.name == "pre":
                _flush_text()
                code_text = _block_text(child)
                if code_text:
                    chunks.append(_make_chunk(code_text, "text", list(heading_path)))
                continue
            # Cokolwiek innego — flat text fallback
            fallback = _normalize_ws(child.get_text(separator=" "))
            if fallback:
                _add_block_to_accumulated(fallback)

        _flush_text()
        return chunks

    # ── Splitter dla potworkowatych tabel ───────────────────────────────────

    def _split_large_table(self, md_table: str, hard_max: int) -> List[str]:
        """Podziel ogromną tabelę markdown na fragmenty zachowując header w każdym.

        Format wejścia:
            | h1 | h2 |
            | --- | --- |
            | row1c1 | row1c2 |
            ...

        Każdy fragment dostaje header + separator + ~N wierszy tak żeby nie
        przekroczyć hard_max.
        """
        lines = md_table.splitlines()
        if len(lines) < 3:
            return [md_table]

        header = lines[0]
        sep = lines[1]
        rows = lines[2:]

        header_size = len(header) + 1 + len(sep) + 1
        out: list[str] = []
        current: list[str] = []
        current_size = header_size
        for row in rows:
            row_size = len(row) + 1
            if current_size + row_size > hard_max and current:
                out.append("\n".join([header, sep, *current]))
                current = []
                current_size = header_size
            current.append(row)
            current_size += row_size
        if current:
            out.append("\n".join([header, sep, *current]))
        return out


# ── Iterator po katalogu cortex_docs ───────────────────────────────────────


def iter_html_files(root: Path) -> Iterable[Path]:
    """Wszystkie pliki .html w drzewie data/cortex_docs/{xdr,xsiam,xsoar,xpanse}/."""
    if not root.exists():
        return
    for product_dir in sorted(root.iterdir()):
        if not product_dir.is_dir():
            continue
        if product_dir.name.startswith(".") or product_dir.name.startswith("_"):
            continue
        if product_dir.name.lower() not in ("xdr", "xsiam", "xsoar", "xpanse"):
            continue
        for f in sorted(product_dir.glob("*.html")):
            yield f

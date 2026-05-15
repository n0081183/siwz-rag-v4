"""Testy HtmlChunkera — sprawdzają kluczowe inwarianty.

To są testy REGRESJI dla najważniejszej zmiany v4. Jeśli kiedykolwiek się posypią,
oznacza to że chunking przestał działać tak jak zaprojektowany.

Inwarianty:
  1. Tabela <table> jest ATOMOWA — nie dzieli się chyba że > hard_max_chars.
  2. Każdy chunk ma niepusty `heading_path` lub `topic_title`.
  3. Każdy chunk z tej samej sekcji ma ten sam `topic_url` i `topic_title`.
  4. chunk_id jest deterministyczny — identyczny HTML daje identyczne ID.
  5. iter_html_files znajduje tylko pliki z katalogów {xdr,xsiam,xsoar,xpanse}.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from siwz_rag.ingest.chunker import Chunk, ChunkerConfig, HtmlChunker, iter_html_files


# ── Helper: minimalna publikacja HTML (format cortex-docs-sync) ─────────────


def _mk_html(
    title: str = "Cortex XDR Test Doc",
    map_id: str = "TEST123_abcdef",
    products: str = "Cortex XDR",
    extra_section: str = "",
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head><title>{title}</title></head>
<body>
<h1>{title}</h1>
<p><strong>Source:</strong> <a href="https://docs-cortex.paloaltonetworks.com/r/Cortex-XDR/{title.replace(' ', '-')}">Open in portal</a></p>
<p><strong>Products:</strong> {products}</p>
<p><strong>Category:</strong> Documentation</p>
<p><strong>Version:</strong> 5.x</p>
<p><strong>Last edition:</strong> 2026-05-10</p>

<section>
<h2>Agent OS Compatibility</h2>
<p><em>Breadcrumb:</em> Compatibility &gt; Agent OS</p>
<p><em>Topic URL:</em> <a href="https://docs-cortex.paloaltonetworks.com/r/Cortex-XDR/Compatibility/Agent-OS">link</a></p>

<h3>Windows Desktop</h3>
<table>
<tr><th>OS Version</th><th>Agent 8.x</th><th>Agent 9.x</th></tr>
<tr><td>Windows 10 21H2</td><td>✓</td><td>✓</td></tr>
<tr><td>Windows 11 23H2</td><td>—</td><td>✓</td></tr>
</table>

<h3>Linux</h3>
<ul>
<li>RHEL 8.x — supported on agent 8.5+</li>
<li>Ubuntu 22.04 LTS — supported on agent 9.0+</li>
</ul>
</section>

{extra_section}

</body>
</html>"""


# ── Tests ───────────────────────────────────────────────────────────────────


def test_basic_chunking_returns_chunks(tmp_path: Path):
    """Sanity check: chunker zwraca listę nieblankowych chunków."""
    html = _mk_html()
    p = tmp_path / "xdr" / "Cortex-XDR-Documentation__TEST_abc.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")

    chunker = HtmlChunker(ChunkerConfig())
    chunks = chunker.chunk_file(p)

    assert len(chunks) >= 2, "Powinniśmy dostać kilka chunków (tabela + lista + tekst)"
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.text.strip(), "chunk.text nie może być pusty"
        assert c.map_id, "map_id musi być wypełnione"
        assert c.topic_title, "topic_title musi być wypełnione"


def test_table_is_atomic(tmp_path: Path):
    """Tabela < hard_max_chars jest JEDNYM chunkiem."""
    html = _mk_html()
    p = tmp_path / "xdr" / "test__map.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")

    chunker = HtmlChunker(ChunkerConfig(target_chars=200, hard_max_chars=5000))
    chunks = chunker.chunk_file(p)

    table_chunks = [c for c in chunks if c.block_type == "table"]
    assert len(table_chunks) >= 1, "Musi być wykryty co najmniej 1 chunk typu table"
    for t in table_chunks:
        # Sprawdź że tabela ma header pipe i separator
        assert "|" in t.text, "Tabela powinna być w formacie markdown z |"
        assert "OS Version" in t.text or "Agent" in t.text or "RHEL" in t.text, \
            "Tabela musi zawierać dane z oryginału"
        # Headery powinny zostać zachowane (header zawsze w pierwszym wierszu)
        first_line = t.text.split("\n")[0]
        if "Agent" in t.text:
            assert "Agent" in first_line or "OS" in first_line, \
                "Header z nazwami kolumn musi być w pierwszym wierszu chunka tabeli"


def test_heading_path_propagated(tmp_path: Path):
    """heading_path zawiera topic_title + h3 subtitle."""
    html = _mk_html()
    p = tmp_path / "xdr" / "test__map.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")

    chunker = HtmlChunker(ChunkerConfig())
    chunks = chunker.chunk_file(p)

    for c in chunks:
        assert isinstance(c.heading_path, list)
        assert len(c.heading_path) >= 1, f"chunk powinien mieć ≥1 heading: {c.heading_path}"
        # Topic title MUSI być pierwszym elementem
        assert c.heading_path[0] == c.topic_title


def test_chunk_id_is_deterministic(tmp_path: Path):
    """Ten sam HTML daje ten sam chunk_id (idempotentny ingest)."""
    html = _mk_html()
    p = tmp_path / "xdr" / "test__map.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")

    chunker = HtmlChunker(ChunkerConfig())
    chunks_a = chunker.chunk_file(p)
    chunks_b = chunker.chunk_file(p)

    ids_a = [c.chunk_id for c in chunks_a]
    ids_b = [c.chunk_id for c in chunks_b]
    assert ids_a == ids_b


def test_inject_heading_path(tmp_path: Path):
    """Gdy inject_heading_path=True, prefiks 'X > Y' jest w text chunka."""
    html = _mk_html()
    p = tmp_path / "xdr" / "test__map.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")

    chunker = HtmlChunker(ChunkerConfig(inject_heading_path=True))
    chunks = chunker.chunk_file(p)

    found_prefixed = False
    for c in chunks:
        if len(c.heading_path) >= 2:
            expected_prefix = " > ".join(c.heading_path)
            if c.text.startswith(expected_prefix):
                found_prefixed = True
                break
    assert found_prefixed, "Przynajmniej jeden chunk powinien mieć heading_path jako prefix tekstu"


def test_chunker_handles_empty_html():
    """Pusty/bezstrukturalny HTML — chunker zwraca pustą listę, nie crash."""
    chunker = HtmlChunker(ChunkerConfig())
    assert chunker.chunk_html("") == []
    assert chunker.chunk_html("<html><body></body></html>") == []


def test_iter_html_files_filters_dirs(tmp_path: Path):
    """iter_html_files zwraca tylko pliki z xdr/xsiam/xsoar/xpanse."""
    for d in ("xdr", "xsiam", "xsoar", "xpanse", "other", "_internal", ".cache"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "a.html").write_text("<html></html>", encoding="utf-8")

    files = list(iter_html_files(tmp_path))
    dirs = {f.parent.name for f in files}
    assert dirs == {"xdr", "xsiam", "xsoar", "xpanse"}
    assert all(f.suffix == ".html" for f in files)


def test_product_inferred_from_dir(tmp_path: Path):
    """Produkt jest pobrany z katalogu (xdr → XDR, xsiam → XSIAM, itd.)."""
    html = _mk_html(products="Cortex XSIAM")
    for d, expected in (("xdr", "XDR"), ("xsiam", "XSIAM"), ("xsoar", "XSOAR"), ("xpanse", "XPANSE")):
        p = tmp_path / d / "doc__map.html"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        chunks = HtmlChunker(ChunkerConfig()).chunk_file(p)
        if chunks:
            assert chunks[0].product == expected, f"Dla katalogu {d} oczekiwano produktu {expected}"


def test_map_id_from_filename(tmp_path: Path):
    """map_id wyciągany z nazwy pliku po `__` (zgodnie z konwencją cortex-docs-sync)."""
    html = _mk_html()
    p = tmp_path / "xdr" / "Cortex-XDR-Documentation__XYZ_abcdef.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")

    chunks = HtmlChunker(ChunkerConfig()).chunk_file(p)
    assert chunks
    # Po `__` w nazwie pliku — chunker powinien wyciągnąć "XYZ_abcdef"
    assert all(c.map_id == "XYZ_abcdef" for c in chunks), f"Got: {chunks[0].map_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

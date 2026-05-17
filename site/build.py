"""
Generate `standalone.html` — a self-contained reader that works by double-click.

Inlines:
- The full markdown guide (LLM_COMPLETE_GUIDE.md) into a <script type="text/plain">.
- The reader CSS and JS so the file does not depend on the assets/ directory.

Run from the project root or from inside site/:
    python site/build.py
    python build.py             # if cwd == site/

The CDN dependencies (marked, KaTeX, Prism) remain external; they require
internet access on first open. This is a deliberate trade-off — embedding them
would add ~600 KB and lock versions inside the bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

THIS_DIR: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = THIS_DIR.parent

GUIDE_PATH: Final[Path] = REPO_ROOT / "LLM_COMPLETE_GUIDE.md"
INDEX_PATH: Final[Path] = THIS_DIR / "index.html"
CSS_PATH: Final[Path] = THIS_DIR / "assets" / "css" / "reader.css"
JS_PATH: Final[Path] = THIS_DIR / "assets" / "js" / "reader.js"
OUT_PATH: Final[Path] = THIS_DIR / "standalone.html"

# Markers in index.html that we replace.
CSS_LINK_MARKER: Final[str] = '<link rel="stylesheet" href="assets/css/reader.css">'
JS_TAG_MARKER: Final[str] = '<script src="assets/js/reader.js"></script>'
INLINE_MD_MARKER: Final[str] = '<script id="inline-markdown" type="text/plain"></script>'


def read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def escape_for_inline_script(text: str) -> str:
    """
    Inside <script type="text/plain">, the only sequence that ends the script
    is the literal '</script>'. Escape any such sequence so the markdown is
    delivered intact to the browser.

    @mitigates StandaloneBundle:Inline against script-tag termination injection
        with </script> escaping in inline markdown payload
    """
    return text.replace("</script>", r"<\/script>")


def build() -> Path:
    html = read_text(INDEX_PATH)
    css = read_text(CSS_PATH)
    js = read_text(JS_PATH)
    md = read_text(GUIDE_PATH)

    if CSS_LINK_MARKER not in html:
        raise SystemExit(f"CSS marker not found in {INDEX_PATH}: {CSS_LINK_MARKER!r}")
    if JS_TAG_MARKER not in html:
        raise SystemExit(f"JS marker not found in {INDEX_PATH}: {JS_TAG_MARKER!r}")
    if INLINE_MD_MARKER not in html:
        raise SystemExit(f"Inline-markdown marker not found in {INDEX_PATH}")

    html = html.replace(CSS_LINK_MARKER, f"<style>\n{css}\n</style>")
    html = html.replace(JS_TAG_MARKER, f"<script>\n{js}\n</script>")
    html = html.replace(
        INLINE_MD_MARKER,
        f'<script id="inline-markdown" type="text/plain">\n{escape_for_inline_script(md)}\n</script>',
    )

    OUT_PATH.write_text(html, encoding="utf-8")

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}  ({size_kb:.1f} KB)")
    print(f"Open by double-click: {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    build()

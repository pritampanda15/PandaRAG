"""Text out of an uploaded PDF."""
import io
import re
from collections import Counter

import pypdf

LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st", "­": "", "\x00": ""})


PAGE_NO = re.compile(r"(?:page\s+)?(\d+)(?:\s+of\s+\d+)?", re.I)
MARGIN = 5


def drop_running_lines(pages):
    """Remove headers, footers and page numbers: lines near the edge of many pages. Only identical lines
    or true page numbers count, so "Patient 1", "Patient 2" headings and table values stay."""
    def key(line, i):
        n = PAGE_NO.fullmatch(line)
        if n:
            return int(n[1]) - i          # constant only when it counts up with the pages
        return line if len(line) > 3 else None

    split = [[l.strip() for l in p.splitlines() if l.strip()] for p in pages]

    def edge(j, lines):
        return j < MARGIN or j >= len(lines) - MARGIN

    seen = Counter()
    for i, lines in enumerate(split):
        seen.update({key(l, i) for j, l in enumerate(lines) if edge(j, lines)} - {None})
    need = max(3, 0.4 * len(pages))       # left and right pages often differ, so half would be too strict
    return ["\n".join(l for j, l in enumerate(lines) if not (edge(j, lines) and seen[key(l, i)] >= need))
            for i, lines in enumerate(split)]


def extract(data):
    """Return {text, pages, blank}; ValueError carries a message fit for the user."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        # Many publisher PDFs are encrypted with an empty password, which opens them for reading.
        locked = reader.is_encrypted and not reader.decrypt("")
        pages = [] if locked else [(p.extract_text() or "").strip() for p in reader.pages]
    except Exception as e:   # pypdf raises many types on malformed files
        raise ValueError(f"could not read this PDF ({type(e).__name__})") from e
    if locked:
        raise ValueError("this PDF is password-protected")
    text = "\n\n".join(p for p in drop_running_lines(pages) if p).translate(LIGATURES)
    # shortcut: a real hyphen at a line end ("dose-\nlimiting") is lost too; needs a dictionary to tell apart
    text = re.sub(r"(?<=[a-z])-[ \t]*\n+[ \t]*(?=[a-z])", "", text)
    # pypdf can emit lone surrogates, which cannot be encoded as JSON
    text = text.encode("utf-8", "ignore").decode("utf-8")
    if not text.strip():
        raise ValueError("no text found: this PDF looks scanned, so run OCR on it first")
    return {"text": text, "pages": len(pages), "blank": sum(not p for p in pages)}

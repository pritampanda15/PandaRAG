"""Text out of an uploaded PDF."""
import io
import re

import pypdf

LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st", "­": "", "\x00": ""})


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
    text = "\n\n".join(p for p in pages if p).translate(LIGATURES)
    # shortcut: a real hyphen at a line end ("dose-\nlimiting") is lost too; needs a dictionary to tell apart
    text = re.sub(r"(?<=[a-z])-[ \t]*\n+[ \t]*(?=[a-z])", "", text)
    # pypdf can emit lone surrogates, which cannot be encoded as JSON
    text = text.encode("utf-8", "ignore").decode("utf-8")
    if not text.strip():
        raise ValueError("no text found: this PDF looks scanned, so run OCR on it first")
    return {"text": text, "pages": len(pages), "blank": sum(not p for p in pages)}

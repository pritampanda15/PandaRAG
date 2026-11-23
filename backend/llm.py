"""LLM access. Uses Claude when ANTHROPIC_API_KEY is set; otherwise callers fall back to extractive mode."""
import os

MODEL = os.getenv("PANDARAG_MODEL", "claude-sonnet-5-5")
_client = None


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def complete(prompt: str, system: str = "", max_tokens: int = 1200) -> str:
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    msg = _client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()

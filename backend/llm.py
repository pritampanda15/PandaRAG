"""LLM access. Uses Claude when ANTHROPIC_API_KEY is set; otherwise callers fall back to extractive mode."""
import os

MODEL = os.getenv("PANDARAG_MODEL", "claude-sonnet-5-5")
_client = None


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def stream(prompt: str, system: str = "", max_tokens: int = 16000):
    """Yield text as it arrives. max_tokens also pays for thinking, which current models do by default,
    so it has to be far larger than the text wanted. Raises if the reply is declined, cut off or empty."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    with _client.messages.stream(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as s:
        yield from s.text_stream
        msg = s.get_final_message()
    if msg.stop_reason != "end_turn":
        raise RuntimeError(msg.stop_reason)
    if not any(b.type == "text" and b.text.strip() for b in msg.content):
        raise RuntimeError("empty")


def complete(prompt: str, system: str = "") -> str:
    return "".join(stream(prompt, system)).strip()

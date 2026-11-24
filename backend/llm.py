"""Multi-provider LLM access: Anthropic, OpenAI-compatible (incl. Ollama/vLLM/OpenRouter), and Gemini.

Zero extra packages required for OpenAI-compatible, Ollama, and Gemini (uses requests + SSE).
Falls back cleanly to extractive mode when no provider is configured.
"""
import json
import os

import requests

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
    "gemini": "gemini-2.5-flash",
}

_anthropic_client = None


def _detect():
    """Return (provider, model, base_url, api_key) from environment."""
    explicit = (os.getenv("PANDARAG_PROVIDER") or "").strip().lower()
    env_model = (os.getenv("PANDARAG_MODEL") or "").strip()

    if explicit in ("ollama",):
        base = (os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        return "ollama", env_model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODELS["ollama"], base, os.getenv("OPENAI_API_KEY") or "ollama"

    if explicit in ("openai", "openai-compatible"):
        base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        key = os.getenv("OPENAI_API_KEY") or ("local" if "api.openai.com" not in base else "")
        return "openai", env_model or DEFAULT_MODELS["openai"], base, key

    if explicit == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        return "gemini", env_model or DEFAULT_MODELS["gemini"], "https://generativelanguage.googleapis.com/v1beta", key

    if explicit == "anthropic":
        return "anthropic", env_model or DEFAULT_MODELS["anthropic"], "https://api.anthropic.com", os.getenv("ANTHROPIC_API_KEY") or ""

    # Auto-detect when PANDARAG_PROVIDER is not explicitly set.
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", env_model or DEFAULT_MODELS["anthropic"], "https://api.anthropic.com", os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        key = os.getenv("OPENAI_API_KEY") or "local"
        prov = "ollama" if "11434" in base else "openai"
        return prov, env_model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODELS[prov], base, key
    if os.getenv("OLLAMA_MODEL"):
        base = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        return "ollama", env_model or os.getenv("OLLAMA_MODEL"), base, "ollama"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return "gemini", env_model or DEFAULT_MODELS["gemini"], "https://generativelanguage.googleapis.com/v1beta", key

    return None, env_model or DEFAULT_MODELS["anthropic"], "", ""


PROVIDER, MODEL, BASE_URL, _API_KEY = _detect()


def info() -> dict:
    prov, model, base, key = _detect()
    return {
        "available": bool(prov and key),
        "provider": prov,
        "model": model if (prov and key) else None,
        "base_url": base if prov in ("openai", "ollama") else None,
    }


def available() -> bool:
    prov, _, _, key = _detect()
    return bool(prov and key)


def label() -> str:
    prov, model, _, _ = _detect()
    return f"{prov}:{model}" if prov else MODEL


def _stream_anthropic(messages: list[dict], system: str, model: str, api_key: str, max_tokens: int):
    global _anthropic_client
    try:
        import anthropic
        if _anthropic_client is None:
            _anthropic_client = anthropic.Anthropic(api_key=api_key)
        with _anthropic_client.messages.stream(
            model=model, max_tokens=max_tokens, system=system, messages=messages,
        ) as s:
            yield from s.text_stream
            msg = s.get_final_message()
        if msg.stop_reason != "end_turn":
            raise RuntimeError(msg.stop_reason)
        if not any(b.type == "text" and b.text.strip() for b in msg.content):
            raise RuntimeError("empty")
        return
    except ImportError:
        pass

    # Pure-requests fallback when the anthropic SDK is not installed.
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages, "stream": True}
    if system:
        payload["system"] = system
    got, stop_reason = False, None
    with requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[5:].strip()
            if not chunk or chunk == "[DONE]":
                continue
            ev = json.loads(chunk)
            if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                txt = ev["delta"].get("text", "")
                if txt:
                    got = True
                    yield txt
            elif ev.get("type") == "message_delta":
                stop_reason = ev.get("delta", {}).get("stop_reason") or stop_reason
    if stop_reason and stop_reason != "end_turn":
        raise RuntimeError(stop_reason)
    if not got:
        raise RuntimeError("empty")


def _stream_openai(messages: list[dict], system: str, model: str, base_url: str, api_key: str, max_tokens: int):
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": msgs, "max_tokens": max_tokens, "stream": True}
    got, finish = False, None
    with requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == "[DONE]":
                break
            ev = json.loads(chunk)
            choices = ev.get("choices") or []
            if not choices:
                continue
            c0 = choices[0]
            delta = (c0.get("delta") or {}).get("content") or ""
            if delta:
                got = True
                yield delta
            if c0.get("finish_reason"):
                finish = c0["finish_reason"]
    if finish == "length":
        raise RuntimeError("max_tokens")
    if not got:
        raise RuntimeError("empty")


def _stream_gemini(messages: list[dict], system: str, model: str, base_url: str, api_key: str, max_tokens: int):
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    payload = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"{base_url}/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    got = False
    with requests.post(url, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[5:].strip()
            if not chunk:
                continue
            ev = json.loads(chunk)
            for cand in ev.get("candidates") or []:
                for part in (cand.get("content") or {}).get("parts") or []:
                    txt = part.get("text") or ""
                    if txt:
                        got = True
                        yield txt
    if not got:
        raise RuntimeError("empty")


def stream(prompt: str, system: str = "", max_tokens: int = 4096, history: list[dict] | None = None):
    """Yield text chunks from the active LLM provider. Raises on failure or empty output."""
    prov, model, base_url, api_key = _detect()
    if not prov or not api_key:
        raise RuntimeError("no LLM provider configured")
    msgs = [{"role": m["role"], "content": m["content"]} for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    msgs.append({"role": "user", "content": prompt})
    if prov == "anthropic":
        yield from _stream_anthropic(msgs, system, model, api_key, max_tokens=max(max_tokens, 4096))
    elif prov in ("openai", "ollama"):
        yield from _stream_openai(msgs, system, model, base_url, api_key, max_tokens=max_tokens)
    elif prov == "gemini":
        yield from _stream_gemini(msgs, system, model, base_url, api_key, max_tokens=max_tokens)
    else:
        raise RuntimeError(f"unsupported provider: {prov}")


def complete(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    return "".join(stream(prompt, system, max_tokens=max_tokens)).strip()

"""Minimal LLM clients used by the spend agent (plain HTTP via `requests`, no vendor SDKs).

  gemini -> Google AI Studio (free key: https://aistudio.google.com/apikey)
  groq   -> Groq Cloud, OpenAI-compatible (free key: https://console.groq.com/keys)
  ollama -> local Ollama server, OpenAI-compatible, no key (https://ollama.com)
"""
import requests

from src import config


class LLMError(RuntimeError):
    pass


class GeminiBackend:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set.")

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature},
        }
        r = requests.post(url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=60)
        if r.status_code != 200:
            raise LLMError(f"Gemini API error {r.status_code}: {r.text[:300]}")
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Gemini response: {r.text[:300]}") from e


class OpenAICompatibleBackend:
    def __init__(self, base_url: str, model: str, api_key: str = "", name: str = "openai-compatible"):
        self.base_url, self.model, self.api_key, self.name = base_url.rstrip("/"), model, api_key, name

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=120)
        except requests.RequestException as e:
            raise LLMError(f"Could not reach {self.name} at {self.base_url}: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"{self.name} API error {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"] or ""


def get_backend(name: str | None = None, api_key: str | None = None, model: str | None = None):
    """Return an LLM client, or None for the offline rule-based mode."""
    name = (name or config.AGENT_BACKEND).lower()
    if name == "rules":
        return None
    if name == "gemini":
        return GeminiBackend(api_key, model)
    if name == "groq":
        key = api_key or config.GROQ_API_KEY
        if not key:
            raise LLMError("GROQ_API_KEY is not set.")
        return OpenAICompatibleBackend("https://api.groq.com/openai/v1", model or config.GROQ_MODEL, key, "groq")
    if name == "ollama":
        return OpenAICompatibleBackend(config.OLLAMA_URL, model or config.OLLAMA_MODEL, "", "ollama")
    raise LLMError(f"Unknown backend '{name}'. Use rules | gemini | groq | ollama.")

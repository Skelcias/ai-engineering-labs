import os
import time

from dataclasses import dataclass, asdict
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# no need for base_url for openai since its provided in the sdk
PROVIDERS = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "model": "gpt-6-luna"},
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "model": "mistral-small-2603",
    },
}


@dataclass
class LLMResult:
    text: str | None
    parsed: BaseModel | None  # l'objet Pydantic validé, si demandé
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_s: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["parsed"] = self.parsed.model_dump() if self.parsed else None
        if self.parsed is not None:
            d["text"] = None
        return d


def complete(
    messages, provider="mistral", model=None, response_model=None
) -> LLMResult:
    client = _get_client(provider)
    model = model or PROVIDERS[provider]["model"]

    start = time.perf_counter()
    if response_model is not None:
        completion = client.chat.completions.parse(
            model=model, messages=messages, response_format=response_model
        )
    else:
        completion = client.chat.completions.create(model=model, messages=messages)
    latency = time.perf_counter() - start
    message = completion.choices[0].message
    return LLMResult(
        text=message.content,
        parsed=getattr(message, "parsed", None),
        provider=provider,
        model=model,
        input_tokens=completion.usage.prompt_tokens,
        output_tokens=completion.usage.completion_tokens,
        reasoning_tokens=completion.usage.completion_tokens_details.reasoning_tokens,
        latency_s=latency,
    )


def _get_client(provider):

    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider : {provider}")

    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["key_env"])
    if api_key is None:
        raise ValueError(f"Missing {cfg["key_env"]} in .env !!!")
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def list_models(provider="mistral") -> list[str]:
    client = _get_client(provider)
    return sorted(m.id for m in client.models.list())

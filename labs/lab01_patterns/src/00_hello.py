import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import complete

Intent = Literal[
    "cancel_order",
    "track_order",
    "change_shipping_address",
    "get_refund",
    "complaint",
    "contact_human_agent",
]
PROVIDERS = ["openai"]


class CustomerRequest(BaseModel):
    intent: Intent
    order_id: int | None = Field(
        description="Order number if explicitly written in the message, else null"
    )
    summary: str


path_file = Path(__file__).parent.parent / "data" / "messages.json"
data = json.loads(path_file.read_text(encoding="utf-8"))
row = next(r for r in data if "order_id" in r["expected"])
messages = [
    {
        "role": "system",
        "content": "You classify customer support messages for an online shop. Identify the intent, extract the order number only if it appears explicitly in the message (otherwise null, never guess), and summarize the request in one sentence.",
    },
    {"role": "user", "content": row["instruction"]},
]

print(f'intent expected : {row["intent"]} & order_id : {row["expected"]["order_id"]}')

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

for provider in PROVIDERS:

    result = complete(
        messages=messages, provider=provider, response_model=CustomerRequest
    )

    print(
        f"[{provider}] intent = {result.parsed.intent} --  order_id = {result.parsed.order_id}  -- summary = {result.parsed.summary}"
    )
    print(
        f"tokens : {result.input_tokens} -> {result.output_tokens} -- latency {result.latency_s:.2f}"
    )
    (results_dir / f"hello_{provider}.json").write_text(
        json.dumps(result.to_dict(), indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

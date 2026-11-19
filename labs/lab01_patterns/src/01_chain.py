import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from common.llm import *

""" Workflow n°1 : Chaining LLM -> Code -> LLM """

PROVIDER = "openai"
REASONING_EFFORT = "medium"
Intent = Literal[
    "cancel_order",
    "track_order",
    "change_shipping_address",
    "get_refund",
    "complaint",
    "contact_human_agent",
]
INTENTS_NEEDING_ID = {
    "cancel_order",
    "track_order",
    "change_shipping_address",
    "get_refund",
}


messages_path = Path(__file__).parent.parent / "data" / "messages.json"
carts_path = Path(__file__).parent.parent / "data" / "carts.json"

data_messages = json.loads(messages_path.read_text(encoding="utf-8"))
carts = json.loads(carts_path.read_text(encoding="utf-8"))
small_data_messages = data_messages[:10]


class CustomerRequest(BaseModel):
    intent: Intent
    order_id: int | None = Field(
        description="Order number if explicitly written in the message, else null"
    )
    summary: str


def extract(message) -> LLMResult:
    messages = [
        {
            "role": "system",
            "content": "You classify customer support messages for an online shop. Identify the intent, extract the order number only if it appears explicitly in the message (otherwise null, never guess), and summarize the request in one sentence.",
        },
        {"role": "user", "content": message},
    ]
    return complete(
        messages=messages,
        provider=PROVIDER,
        response_model=CustomerRequest,
        reasoning_effort=REASONING_EFFORT,
    )


def check(extraction, carts) -> str:
    intent = extraction.parsed.intent
    order_id = extraction.parsed.order_id

    if intent in INTENTS_NEEDING_ID and order_id is None:
        return "need_order_id"
    if order_id is not None and order_id not in carts:
        return "order_not_found"
    return "ok"


def draft_reply(message, extraction, order) -> LLMResult:
    messages = [
        {
            "role": "system",
            "content": order,
        },
        {"role": "user", "content": message},
    ]
    return complete(
        messages=messages,
        provider=PROVIDER,
        reasoning_effort=REASONING_EFFORT,
    )


def run_chain(row, carts) -> dict:
    extraction = extract(row["instruction"])
    status = check(extraction, carts=carts)

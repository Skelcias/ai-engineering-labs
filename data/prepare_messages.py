import json
import re
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
TAG_PATTERN = re.compile(r"\{\{(.+?)\}\}")

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = ds["train"].to_pandas()

### Choose 6 intents & filtered the dataFrame while sampling it

intents = [
    "cancel_order",
    "track_order",
    "change_shipping_address",
    "get_refund",
    "complaint",
    "contact_human_agent",
]

df_filtered = df[df["intent"].isin(intents)]

df_smaller = df_filtered.groupby("intent").sample(n=20, random_state=SEED)

### replace tags by fake values from dummydata "carts" and register col "expected"
## Tags were identified on this sample; the full dataset has 300+ inconsistent tags, mostly noise
SHOP = {
    "Customer Support Hours": "Monday - Friday, 9:00 AM - 6:00 PM",
    "Customer Support Phone Number": "+33 5 56 00 00 00",
    "Customer Support Email": "support@my-awesome-shop.com",
    "Website URL": "https://my-awesome-shop.com",
    "Online Company Portal Info": "https://my-awesome-shop.com",
    "Order Tracking": "'Track my order'",
    "Online Order Interaction": "'Manage my orders'",
    "Purchase History": "'Purchase history'",
    "Order Status": "'Order status'",
}


def draw_row_values(cart, rng):
    return {
        "Order Number": str(cart["id"]),
        "Refund Amount": f'{rng.choice(cart["products"])["discountedTotal"]:.2f}',
        "Currency Symbol": "€",
    }


def replace(text, values):
    def by_value(match):
        name = match.group(1).strip()
        if name not in values:
            raise KeyError(f"Unknown tag : {name}")
        return values[name]

    return re.sub(TAG_PATTERN, by_value, text)


def build_expected(row):
    names = {n.strip() for n in re.findall(TAG_PATTERN, row["instruction"])}
    expected = {}
    if "Order Number" in names:
        expected["order_id"] = int(row["values"]["Order Number"])
    if "Refund Amount" in names:
        expected["refund_amount"] = float(row["values"]["Refund Amount"])
    return expected


def main():
    rng = random.Random(SEED)
    df_final = df_smaller.copy()
    file_path = Path(__file__).parent / "carts.json"
    carts = json.loads(file_path.read_text(encoding="utf-8"))
    df_final["values"] = [
        {**SHOP, **draw_row_values(rng.choice(carts), rng=rng)}
        for _ in range(len(df_final))
    ]
    df_final["expected"] = df_final.apply(build_expected, axis=1)

    df_final["instruction"] = df_final.apply(
        lambda row: replace(row["instruction"], row["values"]), axis=1
    )
    df_final["response"] = df_final.apply(
        lambda row: replace(row["response"], row["values"]), axis=1
    )

    df_final.drop(columns=["values"], inplace=True)

    df_final.to_json(
        Path(__file__).parent / "messages.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    assert not df_final["instruction"].str.contains("{{", regex=False).any()
    assert not df_final["response"].str.contains("{{", regex=False).any()

    df_final["expected"].apply(lambda d: tuple(sorted(d))).value_counts()


if __name__ == "__main__":
    main()

import requests
from pathlib import Path
import json

r = requests.get("https://dummyjson.com/carts?limit=0", timeout=10)
r.raise_for_status()
filtered_data = [
    {
        "id": c["id"],
        "userId": c["userId"],
        "discountedTotal": c["discountedTotal"],
        "products": [
            {
                "title": p["title"],
                "quantity": p["quantity"],
                "discountedTotal": p["discountedTotal"],
            }
            for p in c["products"]
        ],
    }
    for c in r.json()["carts"]
]
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)
file_path = output_dir / "carts.json"

file_path.write_text(
    json.dumps(filtered_data, indent=4, ensure_ascii=False), encoding="utf-8"
)
print(f"Données filtrées sauvegardées avec succès dans : {file_path.resolve()}")
print(len(filtered_data))

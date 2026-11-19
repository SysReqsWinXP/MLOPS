import json

with open("5e-SRD-Monsters.json", "r", encoding="utf-8") as f:
    monsters = json.load(f)

result = []
for m in monsters:
    parts = [m.get("name", ""), f"CR {m.get('challenge_rating', '?')}"]

    if isinstance(m.get("type"), str):
        parts.append(m["type"])
    elif isinstance(m.get("type"), dict):
        parts.append(m["type"].get("name", ""))

    if m.get("armor_class"):
        ac = m["armor_class"][0]
        parts.append(f"AC {ac.get('value', ac) if isinstance(ac, dict) else ac}")
    if m.get("hit_points"):
        parts.append(f"HP {m['hit_points']}")

    for ability in m.get("special_abilities", []):
        parts.append(f"{ability.get('name', '')}: {ability.get('desc', '')}")
    for action in m.get("actions", []):
        parts.append(f"{action.get('name', '')}: {action.get('desc', '')}")

    result.append({
        "index": m.get("index"),
        "name": m.get("name"),
        "rag_text": " ".join(p for p in parts if p)
    })

with open("bestiary.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Обработано {len(result)} монстров")
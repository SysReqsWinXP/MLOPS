import json
import os
import re
from openai import OpenAI

FACTS_FILE = "facts.json"

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

DEFAULT_FACTS = {
    "name": "Unknown",
    "level": 1,
    "weapon": "None",
    "place": "Unknown",
    "health": 100,
}


def load_facts():
    if not os.path.exists(FACTS_FILE):
        with open(FACTS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_FACTS, f, indent=2, ensure_ascii=False)
    with open(FACTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_facts(facts):
    with open(FACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def answer_question(facts: dict, question: str) -> str:
    system_prompt = (
        "You are an assistant that answers questions about a role-playing game character. "
        "You are given a JSON object with the character's facts. "
        "Answer the user's question using ONLY the facts provided. "
        "Be concise. If the answer is not present in the facts, say 'Unknown'. "
        "Reply in the same language as the question."
    )
    user_prompt = (
        f"FACTS (JSON):\n{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"QUESTION:\n{question}"
    )
    return call_llm(system_prompt, user_prompt)


def parse_statement(facts: dict, statement: str) -> str:
    system_prompt = (
        "You are an information-extraction tool. You receive a list of fact NAMES "
        "and a user's STATEMENT. Your job is to detect whether the statement updates "
        "any of the facts. "
        "\n\nOutput format rules (STRICT):\n"
        "- For each fact that the statement updates, output one line in the exact form:\n"
        "  tool_mind(name, \"new_value\")\n"
        "- Use only fact names from the provided list.\n"
        "- The new_value must be exactly what the statement implies, converted to the fact's "
        "natural type (numbers as numbers, strings as strings) but written as plain text inside quotes.\n"
        "- If the statement does not update any fact, output exactly: NOTHING\n"
        "- Do NOT add explanations, comments, or any other text.\n"
        "- Output only tool_mind lines or NOTHING."
    )
    user_prompt = (
        f"FACT NAMES: {json.dumps(list(facts.keys()), ensure_ascii=False)}\n"
        f"CURRENT FACTS: {json.dumps(facts, ensure_ascii=False)}\n\n"
        f"STATEMENT:\n{statement}"
    )
    return call_llm(system_prompt, user_prompt)


TOOL_RE = re.compile(r'tool_mind\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"([^"]*)"\s*\)')


def apply_updates(facts: dict, llm_output: str) -> bool:
    updated = False
    for match in TOOL_RE.finditer(llm_output):
        key, raw_value = match.group(1), match.group(2)
        if key not in facts:
            print(f"[tool] SKIPPED unknown fact '{key}'")
            continue

        current = facts[key]
        new_value = raw_value
        if isinstance(current, int):
            try:
                new_value = int(raw_value)
            except ValueError:
                try:
                    new_value = int(float(raw_value))
                except ValueError:
                    pass
        elif isinstance(current, float):
            try:
                new_value = float(raw_value)
            except ValueError:
                pass

        print(f'[tool] tool_mind({key}, "{raw_value}")')
        facts[key] = new_value
        updated = True
    return updated


def main():
    facts = load_facts()
    print("[facts loaded]", json.dumps(facts, ensure_ascii=False))

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        print(f"[user] {user_input}")

        if user_input.endswith("?"):
            answer = answer_question(facts, user_input)
            print(f"[ai] {answer}")
        else:
            raw = parse_statement(facts, user_input)
            print(f"[ai] {raw}")
            if raw.strip().upper() == "NOTHING" or not TOOL_RE.search(raw):
                print("[tool] NOTHING")
                continue
            if apply_updates(facts, raw):
                save_facts(facts)
                print("[facts saved]", json.dumps(facts, ensure_ascii=False))

if __name__ == "__main__":
    main()
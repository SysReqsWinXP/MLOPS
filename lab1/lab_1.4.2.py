from openai import OpenAI
import re
from datetime import datetime

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

CLASSES = {
    "воин":    "Воин (Fighter): d10 хитов, основная характеристика - Сила. Носит тяжёлую броню, бьёт оружием, получает Second Wind и Action Surge.",
    "маг":     "Маг (Wizard): d6 хитов, основная характеристика - Интеллект. Носит книги заклинаний, готовит заклинания заранее, самая большая палитра заклинаний.",
    "плут":    "Плут (Rogue): d8 хитов, основная характеристика - Ловкость. Sneak Attack, экспертиза в навыках, вороват и скрытен.",
    "жрец":    "Жрец (Cleric): d8 хитов, основная характеристика - Мудрость. Божественная магия, лечит, Channel Divinity, тяжёлая броня у некоторых доменов.",
    "варвар":  "Варвар (Barbarian): d12 хитов, основная характеристика - Сила. Rage, Unarmored Defense, много HP, мало брони.",
}

def get_class(name):
    return CLASSES.get(name.lower().strip(), "Класс не найден.")

SYSTEM_PROMPT = """Ты - ассистент по D&D.

ПРАВИЛО 1. Если пользователь спрашивает про игровой класс персонажа - ответь РОВНО одной строкой, без пояснений:
[TOOL: get_class("название")]

ПРАВИЛО 2. Во всех остальных случаях отвечай обычным текстом, инструмент НЕ вызывай.

Примеры:
Пользователь: Расскажи про мага.
Ты: [TOOL: get_class("маг")]

Пользователь: Что такое плут?
Ты: [TOOL: get_class("плут")]

Пользователь: Привет!
Ты: Привет! Спроси меня про класс D&D - воин, маг, плут, жрец, варвар."""

TOOL_RE = re.compile(
    r'\[\s*TOOL\s*:\s*get_class\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)\s*\]',
    re.IGNORECASE
)

model_name = input("Введите имя модели (например, qwen): ").strip() or "qwen"

default_log = f"chat_{model_name}.log"
log_path = input(f"Файл лога [{default_log}]: ").strip() or default_log

def log(text):
    """Дописывает строку в лог-файл с отметкой времени."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")

log(f"--- СЕССИЯ НАЧАТА (модель: {model_name}) ---")

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

MAX_TOOL_ITERS = 2

print(f"Агент D&D-классов готов (модель: {model_name}). Лог: {log_path}. 'exit' - выход.\n")

while True:
    user_input = input("Вы: ").strip()

    if user_input.lower() in ('exit', 'quit', 'выход'):
        print("Завершение работы.")
        log("--- СЕССИЯ ЗАВЕРШЕНА ---")
        break

    if not user_input:
        continue

    log(f"ПОЛЬЗОВАТЕЛЬ: {user_input}")
    messages.append({"role": "user", "content": user_input})

    tool_iters = 0

    while True:
        print("Ассистент: ", end="", flush=True)
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                temperature=0.0
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print()

            log(f"АССИСТЕНТ: {full_response}")
            messages.append({"role": "assistant", "content": full_response})

            tool_match = TOOL_RE.search(full_response)

            if tool_match and tool_iters < MAX_TOOL_ITERS:
                tool_iters += 1
                arg = tool_match.group(1).strip()
                log(f"СИСТЕМА: get_class('{arg}')")
                print(f"[Система]: get_class('{arg}')")

                result = get_class(arg)

                log(f"СИСТЕМА: Результат -> {result}")
                print(f"[Система]: {result}\n")

                messages.append({
                    "role": "user",
                    "content": (
                        f"[Результат get_class]: {result}\n"
                        "Ответь пользователю кратко по этой информации. "
                        "НЕ вызывай инструмент снова."
                    )
                })
                continue
            else:
                break

        except Exception as e:
            error_msg = f"Ошибка: {e}"
            print(f"\n{error_msg}")
            log(f"ОШИБКА: {error_msg}")
            if messages[-1]["role"] == "user" and "[Результат" not in messages[-1]["content"]:
                messages.pop()
            break
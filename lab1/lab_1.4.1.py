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

SYSTEM_PROMPT = """Ты - ассистент по D&D. Если спрашивают про класс персонажа, ты ДОЛЖЕН вызвать инструмент ровно одной строкой:
[TOOL: get_class("название")]
Пример:
Пользователь: Расскажи про мага.
Ты: [TOOL: get_class("маг")]
В остальных случаях отвечай сам."""

model_name = input("Введите имя модели (например, qwen): ").strip() or "qwen"

default_log = f"chat_{model_name}.log"
log_path = input(f"Файл лога [{default_log}]: ").strip() or default_log

def log(text):
    """Дописывает строку в лог-файл с отметкой времени."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")

log(f"--- СЕССИЯ НАЧАТА (модель: {model_name}) ---")

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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

            tool_match = re.search(r'\[TOOL:\s*get_class\("([^"]+)"\)\]', full_response)

            if tool_match:
                arg = tool_match.group(1)
                log(f"СИСТЕМА: get_class('{arg}')")
                print(f"[Система]: get_class('{arg}')")

                result = get_class(arg)

                log(f"СИСТЕМА: Результат -> {result}")
                print(f"[Система]: {result}\n")

                messages.append({
                    "role": "user",
                    "content": f"[Результат get_class]: {result}\nОтветь пользователю по этой информации."
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
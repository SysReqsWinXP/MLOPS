# -*- coding: utf-8 -*-
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

messages = []

print("Локальный чат готов. Введите 'exit' для выхода.\n")

while True:
    user_input = input("Вы: ")
    
    if user_input.lower() in ['exit', 'quit', 'выход']:
        print("Завершение работы.")
        break
        
    if not user_input.strip():
        continue

    messages.append({"role": "user", "content": user_input})

    print("Ассистент: ", end="", flush=True)
    
    try:
        stream = client.chat.completions.create(
            model="qwen",
            messages=messages,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print()
        
        messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        print(f"\nОшибка: {e}")
        messages.pop()
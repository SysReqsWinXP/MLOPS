import os
from openai import OpenAI
api_key = ""

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

print("AI is ready to work. To exit, type 'exit' or 'quit'.\n")

while True:
    user_question = input("You: ").strip()

    if user_question.lower() in ("exit", "quit"):
        print("Bye!")
        break

    if not user_question:
        continue

    print("AI is thinking...\n")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": user_question}
            ],
            stream=False,
        )
        print(f"DeepSeek: {response.choices[0].message.content}")
        print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {e}")
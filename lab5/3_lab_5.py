import json
import os
import re
from openai import OpenAI

import chromadb
from sentence_transformers import SentenceTransformer

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

MODEL = "model"
PROFILE_FILE = "dnd_profiles.json"

CHROMA_PATH = "./chroma_dnd"
COLLECTION_NAME = "dnd_monsters"
EMBED_MODEL_PATH = "intfloat/e5-small-v2"
RAG_TOP_K = 3
RAG_MIN_SIMILARITY = 0.75


class HybridMemory:
    """Буфер последних k обменов + накопительное резюме старых."""

    def __init__(self, k: int = 4):
        self.k = k
        self.buffer = []
        self.summary = ""

    def _summarize(self, messages: list) -> str:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt = (
            "Сожми следующий фрагмент диалога в краткое резюме (2-4 предложения), "
            "сохранив факты о персонаже игрока, его предпочтения и суть вопросов:\n\n"
            + text
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        if len(self.buffer) > 2 * self.k:
            old = self.buffer[:-self.k]
            self.buffer = self.buffer[-self.k:]
            new_sum = self._summarize(old)
            self.summary = (self.summary + "\n" + new_sum).strip() if self.summary else new_sum

    def as_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[Резюме предыдущего диалога]\n{self.summary}")
        if self.buffer:
            buf = "\n".join(f"{m['role']}: {m['content']}" for m in self.buffer)
            parts.append(f"[Последние сообщения]\n{buf}")
        return "\n\n".join(parts) if parts else "(история пуста)"


class BestiaryRAG:
    """Поиск по бестиарию в Chroma. Модель и клиент грузятся лениво."""

    def __init__(self, path: str = CHROMA_PATH,
                 collection_name: str = COLLECTION_NAME,
                 model_path: str = EMBED_MODEL_PATH,
                 top_k: int = RAG_TOP_K,
                 min_similarity: float = RAG_MIN_SIMILARITY):
        self.path = path
        self.collection_name = collection_name
        self.model_path = model_path
        self.top_k = top_k
        self.min_similarity = min_similarity
        self._model = None
        self._collection = None
        self._available = None

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available

        if not os.path.isdir(self.path):
            print(f"[RAG] Папка {self.path} не найдена — RAG отключён.")
            self._available = False
            return False

        try:
            self._model = SentenceTransformer(self.model_path)
            chroma = chromadb.PersistentClient(path=self.path)
            self._collection = chroma.get_collection(self.collection_name)
        except Exception as e:
            print(f"[RAG] Не удалось загрузить базу: {e}. RAG отключён.")
            self._available = False
            return False

        if self._collection.count() == 0:
            print("[RAG] Коллекция пуста — RAG отключён.")
            self._available = False
            return False

        print(f"[RAG] Загружено {self._collection.count()} монстров.")
        self._available = True
        return True

    def search(self, query: str) -> list[dict]:
        """Возвращает список {'name': ..., 'text': ..., 'similarity': ...}."""
        if not self._ensure_loaded():
            return []

        q_emb = self._model.encode(["query: " + query]).tolist()
        res = self._collection.query(query_embeddings=q_emb, n_results=self.top_k)

        hits = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            sim = 1 - dist
            if sim < self.min_similarity:
                continue
            hits.append({
                "name": meta.get("name", "?"),
                "text": doc,
                "similarity": sim,
            })
        return hits

    @staticmethod
    def as_context(hits: list[dict]) -> str:
        if not hits:
            return ""
        lines = ["[Справочник: монстры из бестиария]"]
        for h in hits:
            lines.append(f"- {h['text']}")
        return "\n".join(lines)


class DnDProfileStore:
    """Хранит имя персонажа, любимые/нелюбимые классы и оружие, уровень, агрессию."""

    def __init__(self, path: str = PROFILE_FILE):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get(self, user_id: str) -> dict:
        return self.data.get(user_id, {
            "character_name": None,
            "level": "beginner",
            "liked_classes": [],
            "disliked_classes": [],
            "liked_weapons": [],
            "disliked_weapons": [],
            "negative_count": 0,
        })

    def update(self, user_id: str, **fields):
        prof = self.get(user_id)
        prof.update({k: v for k, v in fields.items() if v is not None})
        self.data[user_id] = prof
        self._save()
        return prof

    def add_preference(self, user_id: str, kind: str, value: str):
        prof = self.get(user_id)
        if value and value not in prof[kind]:
            prof[kind].append(value)
        self.data[user_id] = prof
        self._save()
        return prof

    def bump_negative(self, user_id: str) -> int:
        prof = self.get(user_id)
        prof["negative_count"] = prof.get("negative_count", 0) + 1
        self.data[user_id] = prof
        self._save()
        return prof["negative_count"]

    def reset_negative(self, user_id: str):
        prof = self.get(user_id)
        prof["negative_count"] = 0
        self.data[user_id] = prof
        self._save()


def classify(question: str) -> dict:
    """Возвращает тему, тональность, уровень, имя персонажа и предпочтения."""
    prompt = f"""Ты анализируешь сообщение игрока в D&D. Верни ТОЛЬКО JSON:
{{
  "topic": "class|weapon|rules|monster|general",
  "sentiment": "positive|negative|neutral",
  "level_hint": "beginner|intermediate|advanced",
  "character_name": null,
  "liked_classes": ["..."],
  "disliked_classes": ["..."],
  "liked_weapons": ["..."],
  "disliked_weapons": ["..."]
}}

Правила:
- topic="monster", если игрок спрашивает про конкретное существо или монстров вообще
  (статы, CR, HP, способности, "есть ли драконы с маленьким здоровьем" и т.п.);
- character_name - имя ПЕРСОНАЖА, если игрок его называет
  ("меня зовут Тралл" -> "Тралл", "мой персонаж - эльфийка Лея" -> "Лея").
  Имя самого игрока (человека) НЕ извлекай.
- в liked_* попадает то, что игрок хвалит или чем любит играть;
- в disliked_* - то, что игрок ругает или не любит;
- если данных нет - используй пустой список [] (а не null).

Сообщение: {question}
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    m = re.search(r"\{.*\}", raw, re.DOTALL)

    empty = {
        "topic": "general", "sentiment": "neutral", "level_hint": "beginner",
        "character_name": None,
        "liked_classes": [], "disliked_classes": [],
        "liked_weapons": [], "disliked_weapons": [],
    }
    if not m:
        return empty

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return empty

    list_keys = ("liked_classes", "disliked_classes", "liked_weapons", "disliked_weapons")
    for key in empty:
        if key not in data or data[key] is None:
            data[key] = empty[key]

    for key in list_keys:
        val = data[key]
        if isinstance(val, str):
            val = [val]
        elif not isinstance(val, list):
            val = []
        data[key] = [v for v in val if isinstance(v, str) and v.strip()]

    for key in ("topic", "sentiment", "level_hint"):
        if not isinstance(data[key], str):
            data[key] = empty[key]

    if not isinstance(data["character_name"], str):
        data["character_name"] = None

    return data


EERIE_MESSAGE = "Вы чувствуете чье-то злобное внимание..."


class DungeonMaster:
    def __init__(self, user_id: str = "test_user_1", k: int = 4, use_rag: bool = True):
        self.user_id = user_id
        self.memory = HybridMemory(k=k)
        self.profiles = DnDProfileStore()
        self.rag = BestiaryRAG() if use_rag else None

    def _system_prompt(self, profile: dict, branch: str, rag_context: str = "") -> str:
        level = profile.get("level", "beginner")
        char_name = profile.get("character_name") or "безымянный герой"
        liked_c = ", ".join(profile.get("liked_classes", [])) or "—"
        disliked_c = ", ".join(profile.get("disliked_classes", [])) or "—"
        liked_w = ", ".join(profile.get("liked_weapons", [])) or "—"
        disliked_w = ", ".join(profile.get("disliked_weapons", [])) or "—"

        base = (
            f"Ты — опытный Dungeon Master и компаньон персонажа по имени {char_name}. "
            f"Уровень игрока: {level}. "
            f"Любимые классы персонажа: {liked_c}; нелюбимые: {disliked_c}. "
            f"Любимое оружие: {liked_w}; нелюбимое: {disliked_w}. "
            "Отвечай в духе фэнтези, кратко (2–5 предложений), давай практичные советы."
        )
        if branch == "negative":
            base += " Игрок раздражён - посочувствуй, предложи выход и поддержи."
        elif branch == "advanced":
            base += " Игрок опытный - используй механику, билды и точные термины."
        elif branch == "beginner":
            base += " Игрок новичок - объясняй простыми словами, без жаргона."

        if rag_context:
            base += (
                "\n\nНиже - выдержки из бестиария, найденные по запросу игрока. "
                "Если вопрос касается монстров, опирайся на эти данные (CR, HP, AC, "
                "способности). Если в выдержках нет ответа — честно скажи об этом.\n\n"
                + rag_context
            )
        return base

    def ask(self, question: str) -> str:
        profile = self.profiles.get(self.user_id)

        cls = classify(question)
        sentiment = cls["sentiment"]
        level_hint = cls["level_hint"]
        topic = cls["topic"]

        char_name = cls.get("character_name")
        if char_name:
            profile = self.profiles.update(self.user_id, character_name=char_name)

        for kind in ("liked_classes", "disliked_classes", "liked_weapons", "disliked_weapons"):
            for value in cls.get(kind, []):
                self.profiles.add_preference(self.user_id, kind, value)

        order = {"beginner": 0, "intermediate": 1, "advanced": 2}
        if order.get(level_hint, 0) > order.get(profile.get("level", "beginner"), 0):
            profile = self.profiles.update(self.user_id, level=level_hint)

        eerie = False
        if sentiment == "negative":
            neg = self.profiles.bump_negative(self.user_id)
        else:
            self.profiles.reset_negative(self.user_id)
            neg = 0

        if neg >= 2:
            branch = "negative"
            eerie = True
        elif profile.get("level") == "advanced":
            branch = "advanced"
        elif profile.get("level") == "beginner":
            branch = "beginner"
        else:
            branch = "default"

        rag_context = ""
        if self.rag is not None and topic == "monster":
            hits = self.rag.search(question)
            rag_context = BestiaryRAG.as_context(hits)
            if hits:
                names = ", ".join(h["name"] for h in hits)
                print(f"[RAG] найдено: {names}")

        history = self.memory.as_context()
        user_msg = f"{history}\n\n[Сообщение игрока]\n{question}"
        messages = [
            {"role": "system", "content": self._system_prompt(profile, branch, rag_context)},
            {"role": "user", "content": user_msg},
        ]

        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.3
        )
        answer = resp.choices[0].message.content.strip()

        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        if eerie:
            return answer + f"\n\n*{EERIE_MESSAGE}*"
        return answer


if __name__ == "__main__":
    dm = DungeonMaster(user_id="test_user_1", k=3)

    dialog = [
        "Привет! Я новичок в D&D, моего героя зовут Нордий, какой класс мне взять?",
        "Обожаю варвара и топоры, а маги с фаерболами это скукота.",
        "Меня убила ловушка! Это ужасно!",
        "Ну сколько можно, второй раз подряд!",
        "Есть ли драконы с маленьким здоровьем?",
        "А что умеет гоблин?",
        "Что ты помнишь о моём персонаже?",
    ]

    for q in dialog:
        print("=" * 70)
        print(f"Игрок: {q}")
        print(f"Мастер: {dm.ask(q)}")

    print("\n=== Профиль персонажа (файл) ===")
    print(json.dumps(dm.profiles.get("test_user_1"), ensure_ascii=False, indent=2))
    print("\n=== Память: резюме ===")
    print(dm.memory.summary or "(пока пусто)")
    print("\n=== Память: буфер ===")
    for m in dm.memory.buffer:
        print(f"  {m['role']}: {m['content'][:80]}...")
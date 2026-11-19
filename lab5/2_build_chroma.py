import json
import chromadb
from sentence_transformers import SentenceTransformer
import os

BESTIARY_FILE = "bestiary.json"
CHROMA_PATH = "./chroma_dnd"
COLLECTION_NAME = "dnd_monsters"
EMBED_MODEL_PATH = "intfloat/e5-small-v2"
BATCH_SIZE = 40


with open(BESTIARY_FILE, "r", encoding="utf-8") as f:
    bestiary = json.load(f)

bestiary = [
    m for m in bestiary
    if m.get("index") and m.get("rag_text")
]
print(f"Монстров для индексации: {len(bestiary)}")


model = SentenceTransformer(EMBED_MODEL_PATH)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


existing_ids = set(collection.get()["ids"])
to_add = [m for m in bestiary if m["index"] not in existing_ids]

print(f"Уже в базе: {len(existing_ids)}")
print(f"Нужно добавить: {len(to_add)}")

if not to_add:
    print("База уже полная, ничего добавлять не нужно.")
else:
    for start in range(0, len(to_add), BATCH_SIZE):
        batch = to_add[start:start + BATCH_SIZE]
        texts = ["passage: " + m["rag_text"] for m in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[m["index"] for m in batch],
            documents=[m["rag_text"] for m in batch],
            embeddings=embeddings,
            metadatas=[{"name": m.get("name", "")} for m in batch],
        )
        print(f"  добавлено {start + len(batch)} / {len(to_add)}")

print(f"\nИтого в коллекции '{COLLECTION_NAME}': {collection.count()} документов")


if collection.count() > 0:
    test_query = "goblin"
    q_emb = model.encode(["query: " + test_query]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=3)

    print(f"\nСамопроверка по запросу '{test_query}':")
    for doc_id, meta, dist in zip(
        res["ids"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        similarity = 1 - dist
        print(f"  [{similarity:.3f}] {meta.get('name', doc_id)} (id={doc_id})")
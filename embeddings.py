# Encode each lore .md file and index with FAISS

import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# 1) Initialize embedding model & FAISS index
embedder = SentenceTransformer("all-mpnet-base-v2")
dim      = embedder.get_sentence_embedding_dimension()
index    = faiss.IndexFlatL2(dim)
meta     = []  # Will hold {"id": int, "text": str}

# 2) Read lore files, embed, and add to index
lore_dir = os.path.join("data", "lore")
for i, fname in enumerate(sorted(os.listdir(lore_dir))):
    if not fname.endswith(".md"):
        continue
    path = os.path.join(lore_dir, fname)
    text = open(path, encoding="utf-8").read().strip()
    vec  = embedder.encode(text)
    index.add(vec.reshape(1, -1))
    meta.append({"id": i, "filename": fname, "text": text})

# 3) Persist index and metadata
os.makedirs("data", exist_ok=True)
faiss.write_index(index, "data/lore.index")
with open("data/lore_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Indexed {len(meta)} lore documents.")

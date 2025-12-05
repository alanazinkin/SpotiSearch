# query_data.py - Query pre-computed embeddings
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_ID = "mmarkusmalone/album_moods_embedding_stage2"

# Load model and pre-computed embeddings
print("Loading model...")
model = SentenceTransformer(MODEL_ID)

print("Loading pre-computed embeddings from embeddings.npy...")
embeddings = np.load("building_embedding_data/embeddings.npy")  # shape (N, D)

print("Loading metadata from embeddings_metadata.csv...")
df_emb = pd.read_csv("building_embedding_data/embeddings_metadata.csv")

print(f"Loaded {len(df_emb)} albums with embeddings of shape {embeddings.shape}")

def l2_normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    # avoid division by zero
    norm[norm == 0] = 1.0
    return v / norm

# normalize stored embeddings once
embeddings_norm = l2_normalize(embeddings.astype(np.float32))

def query_vibe_console(vibe_text, top_k=5):
    # encode query with same model
    q = model.encode([vibe_text], convert_to_numpy=True)[0].astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)  # normalize query
    # cosine similarity = dot(normalized_vectors, normalized_query)
    sims = embeddings_norm.dot(q)  # shape (N,)
    # top-k indices (descending)
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        row = df_emb.iloc[idx]
        results.append({
            "rank": rank,
            "album_id": row["album_id"],
            "artist": row.get("artist", ""),
            "title": row.get("title", ""),
            "genre": row.get("genre", ""),
            "year_released": row.get("year_released", ""),
            "review": row.get("review", "")[:500],  # first 100 chars
            "score": float(sims[idx])
        })
    return results

# Example usage:
# Example usage:
if __name__ == "__main__":
    queries = [
        "Electronic french trance"
    ]
    for vibe in queries:
        print(f"\n=== Query: '{vibe}' ===")
        top5 = query_vibe_console(vibe, top_k=5)
        for r in top5:
            print(f'{r["rank"]}. {r["artist"]} — {r["title"]} ({r["year_released"]}) score={r["score"]:.4f}')
            print(f'   Review: {r["review"]}...')
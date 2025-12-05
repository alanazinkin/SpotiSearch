# encode_albums.py
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import pandas as pd
from tqdm import tqdm
import numpy as np

# Log in to Hugging Face (uses stored token if available, or prompts interactively)
login()

MODEL_ID = "mmarkusmalone/album_moods_embedding_stage2"

# 1) load CSV
df = pd.read_csv("building_embedding_data/albums_in_both.csv")  # columns: artist,title,year_released,rating,review,genre,query,album_id
print(f"Loaded {len(df)} albums for embedding.")
# 2) build text field you want to embed (tweak to taste)
def build_text(row):
    # include artist/title/genre/year and the review text; keep length reasonable
    parts = []
    if pd.notna(row.get("artist")):
        parts.append(f"{row['artist']} — {row['title']}")
    else:
        parts.append(str(row['title']))
    if pd.notna(row.get("genre")):
        parts.append(f"Genre: {row['genre']}")
    if pd.notna(row.get("year_released")):
        parts.append(f"Released: {int(row['year_released'])}")
    if pd.notna(row.get("rating")):
        parts.append(f"Rating: {row['rating']}")
    if pd.notna(row.get("review")):
        parts.append(f"Review: {row['review']}")
    return ". ".join(parts)

df["text_for_embedding"] = df.apply(build_text, axis=1)
print("Built text for embedding.")
# 3) load embedding model
model = SentenceTransformer(MODEL_ID)
print("Loaded embedding model.")
# 4) batch-encode
batch_size = 64
embeddings = []
for i in tqdm(range(0, len(df), batch_size)):
    batch = df["text_for_embedding"].iloc[i:i+batch_size].tolist()
    emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
    embeddings.append(emb)
embeddings = np.vstack(embeddings)  # shape (N, 768)

# Save embeddings as .npy and metadata as .csv
np.save("embeddings.npy", embeddings)
print("Saved embeddings to embeddings.npy")

# Save metadata (all columns except the original text_for_embedding)
df_meta = df.drop(columns=["text_for_embedding"], errors="ignore")
df_meta.to_csv("embeddings_metadata.csv", index=False)
print("Saved metadata to embeddings_metadata.csv")

print("Embeddings shape:", embeddings.shape)
import torch.nn.functional as F
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.song_src.config.config import base_text_model, device, tok
from src.song_src.engine.data_utils import generate_text_for_embedding
from models.song_model.text_to_features_model import TextToSpotifyFeatures

BATCH_SIZE = 32

def load_embeds(path: str, device: torch.device) -> torch.Tensor | None:
    """
    Loads normalized feature vectors (a dictionary of vectors) from a .pkl file,
    sorts them by index, and converts them to a PyTorch tensor.
    """
    try:
        with open(path, 'rb') as f:
            song_embeds_dict = pickle.load(f)

        print(f"Loaded {len(song_embeds_dict)} embeddings from {path}.")
        sorted_indices = sorted(song_embeds_dict.keys())
        embeddings_list = [song_embeds_dict[idx] for idx in sorted_indices]

        song_embeds_tensor = torch.tensor(
            np.array(embeddings_list),
            dtype=torch.float32,
            device=device
        )

        return song_embeds_tensor

    except FileNotFoundError:
        print(f"Error: Embeddings file not found at {path}.")
        return None
    except Exception as e:
        print(f"Error loading or processing embeddings: {e}")
        return None

def compute_all_embeddings(spot_model, df, tok, device) -> dict:
    # Get combined text strings
    texts = generate_text_for_embedding(df)

    # Extract input text features
    song_text_to_embeds = {}

    spot_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Computing Embeddings"):
            batch_texts = texts[i:i + BATCH_SIZE]

            enc = tok(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(device)

            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]

            # Use spot_model to get final feature vector
            # The TextToSpotifyFeatures model handles the internal text model embedding
            embeddings = spot_model(input_ids, attn)

            # Normalize for cosine similarity
            embeddings = F.normalize(embeddings, dim=-1)

            # Map the embeddings back to the original DataFrame index
            for j, emb in enumerate(embeddings.cpu().numpy()):
                # Use the DataFrame's original index as the key
                original_df_index = df.index[i + j]
                song_text_to_embeds[original_df_index] = emb.tolist()  # Store as list for pickle/JSON

    return song_text_to_embeds

def save_new_embeddings(spot_model, df, tok, device, path):
    print("Recomputing ALL embeddings now...")
    new_embeds_dict = compute_all_embeddings(spot_model, df, tok, device)
    with open(path, 'wb') as f:
        pickle.dump(new_embeds_dict, f)
    print("Embeddings saved to disk.")

def main():
    DF_PATH = "song_mood_final/data/spotify_all_songs_with_review_cols_updated.csv"
    EMBEDDINGS_OUTPUT_PATH = "song_mood_final/data/song_embeddings.pkl"

    df = pd.read_csv(DF_PATH)

    feature_cols = [
        "energy", "danceability", "valence", "tempo",
        "loudness", "acousticness", "liveness",
        "speechiness", "instrumentalness",
        "time_signature", "mode", "key"
    ]
    out_dim = len(feature_cols)
    spot_model = TextToSpotifyFeatures(base_text_model, out_dim=out_dim).to(device)
    state = torch.load("../../../models/song_model/spotify_model_weights.pth", map_location=device)
    spot_model.load_state_dict(state)
    spot_model.eval()

    # Compute and Save
    song_embeds_dict = compute_all_embeddings(spot_model, df, tok, device)

    with open(EMBEDDINGS_OUTPUT_PATH, 'wb') as f:
        pickle.dump(song_embeds_dict, f)
    print(f"Successfully computed and saved {len(song_embeds_dict)} embeddings to {EMBEDDINGS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
import torch
import torch.nn.functional as F
from gemini_description_generator import generate_descriptions_for_indices
import pandas as pd
import numpy as np


class SearchEngine:
    def __init__(
        self,
        spot_model,          # TextToSpotifyFeatures (text -> spotify feature space)
        df,                  # pandas DataFrame with track metadata
        song_embeds,         # torch.Tensor of shape (N, d)
        tok,                 # tokenizer from config
        device,              # torch.device from config
        text_embed_model,    # base_text_model for reranking (semantic text space)
    ):
        self.spot_model = spot_model
        self.df = df
        self.song_embeds = song_embeds
        self.tok = tok
        self.device = device
        self.text_embed_model = text_embed_model

        if not isinstance(self.song_embeds, torch.Tensor):
            self.song_embeds = torch.tensor(self.song_embeds, dtype=torch.float32)
        self.song_embeds = self.song_embeds.to(self.device)

    import pandas as pd
    import numpy as np

    def update_dataframe(self, candidate_indices, descriptions):

        if "small_text" not in self.df.columns:
            print("Error: DataFrame must contain a 'small_text' column.")
            return

        # Convert the list of indices (which are POSITIONS) to a NumPy array for .iloc
        candidate_pos_array = np.array(candidate_indices)

        # Get the position of the 'small_text' column
        small_text_col_pos = self.df.columns.get_loc("small_text")

        # 1. RETRIEVE CURRENT VALUES using ILOC (accesses by integer position)
        # This retrieves a Series containing the current small_text values for the candidate positions.
        current_values = self.df.iloc[candidate_pos_array, small_text_col_pos]

        # 2. Identify missing values
        # Use .values to work with the boolean mask for indexing arrays
        is_missing = current_values.isna().values | (current_values.astype(str).str.strip().str.len().values == 0)

        # Get the POSITIONS that need updating
        positions_to_update = candidate_pos_array[is_missing]

        # Get the corresponding descriptions that need to be assigned
        descriptions_to_assign = np.array(descriptions)[is_missing]

        print(f"Indices/Positions to Update: {positions_to_update.tolist()}")

        # 3. ASSIGN NEW VALUES using ILOC
        # This assigns the new descriptions to the correct integer positions in the DataFrame
        self.df.iloc[positions_to_update, small_text_col_pos] = descriptions_to_assign

    def _save_df(self, path: str = "data/spotifyData/spotify_all_songs_with_review_cols_updated.csv"):
        print(f"Attempting to save DF to: {path}")
        self.df.to_csv(path, index=False)
        print(f"Successfully wrote updated reviews to: {path}")

    def encode_query_to_feature_vec(self, query: str) -> torch.Tensor:
        """
        Encode user text query into the same feature space as song_embeds
        using the trained spot_model.
        Returns a 1D vector of shape (d,) on self.device.
        """
        enc = self.tok(
            query,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64,
        ).to(self.device)

        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        self.spot_model.eval()
        with torch.no_grad():
            pred = self.spot_model(input_ids, attn)  # (1, d) if your model is defined that way

        pred = pred.squeeze(0)  # (d,)
        return pred

    def embed_texts(self, texts, max_length=128, batch_size=32):
        """
        Embed free-form text using the base_text_model (semantic space).
        Returns a tensor of shape (len(texts), d).
        """
        model = self.text_embed_model
        tok = self.tok

        all_embs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                enc = tok(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                out = model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1)
                x = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                x = F.normalize(x, dim=-1)
                all_embs.append(x.cpu())

        return torch.cat(all_embs, dim=0)  # (len(texts), d)

    # ---------- Main search ----------

    def search_songs(self, query: str, k: int = 10, rerank_factor: int = 2):
        """
        Returns a list of dicts:
        {
            "track_id": ...,
            "name": ...,
            "artist": ...,
            "score": ...,
        }
        """
        results = []

        # ---------- Stage 1: base retrieval in Spotify-feature space ----------

        q_vec = self.encode_query_to_feature_vec(query)          # (d,)
        q_vec = F.normalize(q_vec, dim=-1)                       # (d,)
        sims_spotify = self.song_embeds @ q_vec                  # (N,)

        n_candidates = min(rerank_factor * k, self.song_embeds.shape[0])
        top_vals, top_idx = torch.topk(sims_spotify, n_candidates)

        # Path A: Skip Rerank (k >= 30)
        if k >= 30:
            return self.append_top_songs(k, results, top_idx, top_vals, set())

        # ---------- Stage 2: Rerank with Gemini descriptions + base_text_model ----------

        candidate_indices = top_idx.tolist()
        descriptions = generate_descriptions_for_indices(candidate_indices, self.df, 10)
        print(candidate_indices)

        # Fill in missing small_text descriptions in df  with the new LLM descriptions
        self.update_dataframe(candidate_indices, descriptions)
        self._save_df()

        # 2. Embed query + descriptions with base_text_model
        query_desc_emb = self.embed_texts([query])[0]  # (d,)
        desc_embs = self.embed_texts(descriptions)     # (n_candidates, d)

        # 3. Cosine sim in text space
        sims_text = desc_embs @ query_desc_emb         # (n_candidates,)
        sorted_scores, sorted_idx_local = torch.sort(sims_text, descending=True)
        reranked_global_indices = [candidate_indices[i] for i in sorted_idx_local.tolist()]

        results = self.append_top_songs(
            k=k,
            results=results,
            indices=torch.tensor(reranked_global_indices),
            scores=sorted_scores,
            seen_ids=set(),
        )

        return results

    def append_top_songs(self, k: int, results: list,indices, scores, seen_ids: set) -> list:
        """
        Generic helper to append top songs to `results` given indices + scores.

        - If `seen_ids` is provided, deduplicates by track_id.
        - Stops when `len(results) >= k`.
        - Returns the (possibly) extended `results` list.
        """
        for score, idx in zip(scores.tolist(), indices.tolist()):
            row = self.df.iloc[idx]
            track_id = row.get("track_id", idx)

            if track_id in seen_ids:
                continue
            seen_ids.add(track_id)

            results.append(
                {
                    "track_id": track_id,
                    "name": row["track_name"],
                    "artist": row["track_artist"],
                    "score": float(score),
                }
            )

            if len(results) >= k:
                break

        return results
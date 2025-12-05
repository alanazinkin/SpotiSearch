import torch
import torch.nn.functional as F
from song_mood_final.engine.gemini_description_generator import generate_descriptions_for_indices
import numpy as np

class SearchEngine:
    def __init__(
            self,
            spot_model,  # TextToSpotifyFeatures (text -> spotify feature space)
            df,  # pandas DataFrame with track metadata
            song_embeds,  # torch.Tensor of shape (N, d) - pre-computed
            tok,  # Tokenizer from config
            device,  # torch.device from config
            text_embed_model,  # base_text_model for reranking (semantic text space)
    ):
        self.spot_model = spot_model
        self.df = df
        self.song_embeds = song_embeds
        self.tok = tok
        self.device = device
        self.text_embed_model = text_embed_model


    '''AI assisted in generating the update_dataframe method'''
    def update_dataframe(self, candidate_indices, descriptions):
        if "gemini_review" not in self.df.columns:
            print("Error: DataFrame must contain a 'gemini_review' column.")
            return

        candidate_pos_array = np.array(candidate_indices)
        gemini_review_col_pos = self.df.columns.get_loc("gemini_review")
        current_values = self.df.iloc[candidate_pos_array, gemini_review_col_pos]
        is_missing = current_values.isna().values | (current_values.astype(str).str.strip().str.len().values == 0)

        positions_to_update = candidate_pos_array[is_missing]
        descriptions_to_assign = np.array(descriptions)[is_missing]

        if positions_to_update.size > 0:
            print(f"In-memory DF update: Assigning {positions_to_update.size} new reviews.")
            self.df.iloc[positions_to_update, gemini_review_col_pos] = descriptions_to_assign
        else:
            print("No new reviews generated for candidates.")

    def _save_df(self, path: str = "song_mood_final/data/spotify_all_songs_with_review_cols_updated.csv"):
        print(f"Attempting to save DF to: {path}")
        self.df.to_csv(path, index=False)
        print(f"Successfully wrote updated reviews to: {path}")

    def encode_query_to_feature_vec(self, query: str) -> torch.Tensor:
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
            pred = self.spot_model(input_ids, attn).squeeze(0)
        return pred

    def embed_texts(self, texts, max_length=128, batch_size=32):
        model = self.text_embed_model
        tok = self.tok

        all_embs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i: i + batch_size]
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

        return torch.cat(all_embs, dim=0)

    def search_songs(self, query: str, k: int = 10, rerank_factor: int = 2):
        results = []

        # Stage 1: base retrieval in Spotify-feature space
        q_vec = self.encode_query_to_feature_vec(query)
        q_vec = F.normalize(q_vec, dim=-1)
        sims_spotify = self.song_embeds @ q_vec

        n_candidates = min(rerank_factor * k, self.song_embeds.shape[0])
        top_vals, top_idx = torch.topk(sims_spotify, n_candidates)

        # Path A: Skip Rerank
        if k >= 30:
            return self.append_top_songs(k, results, top_idx, top_vals, set())

        # Stage 2: Rerank with Gemini descriptions + base_text_model

        candidate_indices = top_idx.tolist()
        descriptions = generate_descriptions_for_indices(candidate_indices, self.df, 10)
        print(candidate_indices)

        self.update_dataframe(candidate_indices, descriptions)
        self._save_df()

        # 2. Embed query + descriptions with base_text_model
        query_desc_emb = self.embed_texts([query])[0]
        desc_embs = self.embed_texts(descriptions)

        # 3. Cosine sim in text space
        sims_text = desc_embs @ query_desc_emb
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

    def append_top_songs(self, k: int, results: list, indices, scores, seen_ids: set) -> list:
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
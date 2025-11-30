import torch
import torch.nn.functional as F
from gemini_description_generator import generate_descriptions_for_indices


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

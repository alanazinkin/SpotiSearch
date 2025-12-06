import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

MODEL_ID_DEFAULT = "mmarkusmalone/album_moods_embedding_stage2"
EMBEDS_REL_PATH = "album_mood_final/backend/building_embedding_data/embeddings.npy"
META_REL_PATH = "album_mood_final/backend/building_embedding_data/embeddings_metadata.csv"

def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return v / norm

class AlbumVibeSearcher:
    def __init__(
        self,
        model_id: str = MODEL_ID_DEFAULT,
        embeddings_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        use_verbose: bool = False,
    ):
        # Resolve default paths if not provided
        if embeddings_path is None:
            embeddings_path = EMBEDS_REL_PATH
        if metadata_path is None:
            metadata_path = META_REL_PATH

        self.model_id = model_id
        self.embeddings_path = str(Path(embeddings_path))
        self.metadata_path = str(Path(metadata_path))
        self.use_verbose = use_verbose

        if self.use_verbose:
            print(f"[AlbumVibeSearcher] Loading model '{self.model_id}'")
        self.model = SentenceTransformer(self.model_id)

        if self.use_verbose:
            print(f"[AlbumVibeSearcher] Loading embeddings from '{self.embeddings_path}'")
        self.embeddings = np.load(self.embeddings_path).astype(np.float32)  # shape (N, D)

        if self.use_verbose:
            print(f"[AlbumVibeSearcher] Loading metadata from '{self.metadata_path}'")
        self.df_emb = pd.read_csv(self.metadata_path).reset_index(drop=True)

        if len(self.df_emb) != len(self.embeddings):
            min_len = min(len(self.df_emb), len(self.embeddings))
            if self.use_verbose:
                print(
                    f"[AlbumVibeSearcher] WARNING: metadata rows ({len(self.df_emb)}) "
                    f"!= embeddings rows ({len(self.embeddings)}). Truncating to {min_len}."
                )
            self.df_emb = self.df_emb.iloc[:min_len].reset_index(drop=True)
            self.embeddings = self.embeddings[:min_len]

        self.embeddings_norm = l2_normalize(self.embeddings)

        if self.use_verbose:
            print(f"[AlbumVibeSearcher] Initialized with {len(self.df_emb)} items.")

    def query(self, vibe_text: str, top_k: int = 5) -> List[Dict]:
        """Encode vibe_text and return top_k results"""
        # encode query
        q = self.model.encode([vibe_text], convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)  # normalize query vector

        sims = self.embeddings_norm.dot(q)  # shape (N,)
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            row = self.df_emb.iloc[idx]
            results.append({
                "rank": rank,
                "album_id": row.get("album_id", ""),
                "artist": row.get("artist", ""),
                "title": row.get("title", ""),
                "genre": row.get("genre", ""),
                "year_released": row.get("year_released", ""),
                "review": str(row.get("review", ""))[:500],
                "score": float(sims[idx]),
            })
        return results

_default_searcher: Optional[AlbumVibeSearcher] = None

def init_default_searcher(force_reload: bool = False, **kwargs) -> AlbumVibeSearcher:
    global _default_searcher
    if _default_searcher is None or force_reload:
        _default_searcher = AlbumVibeSearcher(**kwargs)
    return _default_searcher

def query_vibe_console(vibe_text: str, top_k: int = 5) -> List[Dict]:
    """Old API kept for compatibility: uses a module-level cached searcher."""
    searcher = init_default_searcher()
    return searcher.query(vibe_text, top_k=top_k)

# If run as script, do a simple smoke test (won't run on import in production if not __main__).
if __name__ == "__main__":
    s = init_default_searcher(use_verbose=True)
    print(s.query("dreamy nostalgic indie folk", top_k=5))

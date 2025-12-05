import time
import os
import streamlit as st

# Attempt to import album search backend (may need correct path)
try:
    from album_mood_final.backend.query_data import query_vibe_console
    _HAS_ALBUM_BACKEND = True
except Exception:
    _HAS_ALBUM_BACKEND = False

# Attempt to import spotify/song search engine pieces
try:
    import torch
    from song_mood_final.engine.compute_embeddings import load_embeds
    from song_mood_final.engine.search_engine import SearchEngine
    from song_mood_final.config.config import tok, device, base_text_model, load_config, save_config
    from song_mood_final.models.text_to_features_model import TextToSpotifyFeatures
    from song_mood_final.models.train_model import load_dataframe
    _HAS_SONG_BACKEND = True
except Exception:
    _HAS_SONG_BACKEND = False

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Mood-Based Music Collection Searcher",
    page_icon="🎧",
    layout="centered"
)

# ========== CUSTOM CSS ==========
SCANDINAVIAN_CSS = """
<style>
:root{
  --bg: #e0f4ff;
  --navy: #040a29;
  --chip-border: #e6e1d8;
  --card-bg: #ffffff;
}

body, .stApp {
  background-color: var(--bg) !important;
  color: var(--navy) !important;
  font-family: 'Inter', sans-serif;
}

h1, h2, h3, label, p, div {
  color: var(--navy) !important;
}

.sidebar {
    color: pink;
}
.result-card {
  padding: 0.9rem;
  border-radius: 10px;
  background: var(--card-bg);
  border: 1px solid #dedacf;
  margin-bottom: 0.9rem;
}

.result-title { font-size:1rem; font-weight:600; color:var(--navy); }
.result-meta { font-size:0.85rem; color:#556; }
.result-review { margin-top:0.35rem; font-size:0.85rem; color:var(--navy); }

/* input styling */
.stTextInput>div>div>input {
  background-color: white !important;
  border-radius: 8px !important;
  border: 1px solid #ccc !important;
  padding: 0.6rem !important;
  color: var(--navy) !important;
}

/* primary search button */
.primary-search-button button {
  background-color: var(--navy) !important;
  color: var(--bg) !important;
  border-radius: 8px !important;
  padding: 0.5rem 0.9rem !important;
  border: none !important;
  font-weight: 600;
}

/* minor responsive tweak */
@media (max-width:640px) {
  .result-title { font-size:0.98rem; }
  .result-meta, .result-review { font-size:0.82rem; }
}
</style>
"""
st.markdown(SCANDINAVIAN_CSS, unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("<h1 style='text-align:center; margin-top:18px;'>🎧 Mood-Based Music Collection Searcher</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; margin-bottom:6px;'>Use the tabs to choose Album-level or Spotify song-level vibe search.</p>",
    unsafe_allow_html=True
)

# Short explanatory note requested:
st.markdown(
    "<p style='text-align:center; margin-bottom:12px;'><em>Note:</em> the Album search uses a fine-tuned embedding model, while the Spotify/song search uses an MLP (Text→Spotify features) front-end that reranks candidates using learned Spotify features + embeddings.</p>",
    unsafe_allow_html=True
)

# ========== TABS FOR TWO SEARCH FLOWS ==========
tab_album, tab_spotify = st.tabs(["Album Vibe Search", "Spotify Vibe Search"])

# ---------------- ALBUM TAB ----------------
with tab_album:
    st.subheader("Album Vibe Search")
    if not _HAS_ALBUM_BACKEND:
        st.warning(
            "Album backend couldn't be imported. Make sure `backend.query_data.query_vibe_console` is available "
            "in your project path."
        )

    q_album = st.text_input("Enter vibe text (albums):", placeholder="e.g., dreamy nostalgic indie folk", key="album_query")
    top_k = st.slider("Results to return", 5, 20, 5, key="album_topk")

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="primary-search-button">', unsafe_allow_html=True)
        btn_album_search = st.button("Search Albums")
        st.markdown('</div>', unsafe_allow_html=True)

    if btn_album_search:
        q_text = (q_album or "").strip()
        if not q_text:
            st.warning("Please enter a query (describe the vibe you want).")
        else:
            if not _HAS_ALBUM_BACKEND:
                st.error("Album search backend not available. Cannot run search.")
            else:
                with st.spinner("Searching for matching albums..."):
                    try:
                        results = query_vibe_console(q_text, top_k=top_k)
                    except Exception as e:
                        st.error(f"Error running album search: {e}")
                        results = []

                st.markdown("---")
                st.subheader(f"Top {len(results)} Results for: «{q_text}»")
                if not results:
                    st.info("No results found.")
                for r in results:
                    artist = r.get("artist", "") or ""
                    title = r.get("title", "") or ""
                    genre = r.get("genre", "") or ""
                    year = r.get("year_released", "") or ""
                    score = r.get("score", 0.0) or 0.0
                    review = r.get("review", "") or ""
                    st.markdown(
                        f"""
                        <div class='result-card'>
                            <div class='result-title'>{r.get("rank","")}. {artist} — {title}</div>
                            <div class='result-meta'>{genre} • {year} • score={score:.4f}</div>
                            <div class='result-review'>{review}...</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# ---------------- SPOTIFY TAB ----------------
with tab_spotify:
    st.subheader("Spotify Vibe Search")
    if not _HAS_SONG_BACKEND:
        st.warning(
            "Spotify/song backend couldn't be fully imported. Verify the `song_mood_final` package, model weights, and config are present."
        )

    # Sidebar controls (kept in sidebar to match original)
    st.sidebar.header("Search Settings")
    rerank_factor = st.sidebar.slider(
        "Rerank factor (candidate pool size)",
        min_value=1,
        max_value=3,
        value=2,
        step=1,
        help=(
            "First retrieve k * rerank_factor candidates with the Spotify-feature model, "
            "then rerank them using Gemini-generated descriptions (for k < 30)."
        ),
    )
    show_debug = st.sidebar.checkbox("Show debug info (track IDs & raw scores)", value=False)

    @st.cache_resource
    def get_engine():
        if not _HAS_SONG_BACKEND:
            return None

        # 1. Load data (df)
        config = load_config()
        filePath = config.get('songs_file_path')
        df = load_dataframe(filePath)

        # 2. Recreate model architecture
        feature_cols = [
            "energy", "danceability", "valence", "tempo",
            "loudness", "acousticness", "liveness",
            "speechiness", "instrumentalness",
            "time_signature", "mode", "key"
        ]
        out_dim = len(feature_cols)
        spot_model = TextToSpotifyFeatures(base_text_model, out_dim=out_dim).to(device)

        # 3. Load trained weights
        state_path = "song_mood_final/models/spotify_model_weights.pth"
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Required weights file missing at {state_path}.")
        state = torch.load(state_path, map_location=device)
        spot_model.load_state_dict(state)
        spot_model.eval()

        # 4. Build SearchEngine
        embeddings_path = config.get('embeddings_file_path')
        song_embeds_tensor = load_embeds(embeddings_path, device)

        if song_embeds_tensor is None:
            raise FileNotFoundError(f"Required embeddings file missing at {embeddings_path}.")

        search_engine = SearchEngine(
            spot_model=spot_model,
            df=df,
            song_embeds=song_embeds_tensor,
            tok=tok,
            device=device,
            text_embed_model=base_text_model,
            path=filePath
        )
        return search_engine

    # Initialize engine (lazy)
    engine = None
    try:
        engine = get_engine()
    except Exception as e:
        # show friendly error but allow the UI to continue
        st.error(f"Could not initialize search engine: {e}")
        engine = None

    query_song = st.text_input(
        "Describe the vibe (songs):",
        placeholder="e.g., upbeat summer roadtrip pop with female vocals",
        key="song_query",
    )
    k = st.slider("Number of songs (k)", min_value=1, max_value=50, value=10, step=1, key="song_k")
    go = st.button("Search Songs")

    # run-count helper (copied from original)
    def increment_run_count():
        try:
            config = load_config()
            current_run_count = config.get('run_count', 0)
            max_runs = config.get('max_runs_before_recompute', 100)
            new_run_count = current_run_count + 1
            config['run_count'] = new_run_count
            save_config(config)
            display_run = new_run_count if new_run_count <= max_runs else max_runs
            st.sidebar.markdown(f"**Run Count:** {display_run} / {max_runs}")
            if new_run_count >= max_runs:
                st.sidebar.warning(
                    f"**{max_runs} RUNS REACHED!** \nIt is recommended to re-train the model to capture new trends.")
        except Exception:
            # non-fatal; ignore
            pass

    if go:
        if not query_song.strip():
            st.warning("Please enter a description of the vibe before searching.")
        else:
            if engine is None:
                st.error("Search engine not available. Cannot run Spotify search.")
            else:
                with st.spinner("Searching…"):
                    try:
                        t1 = time.time()
                        results = engine.search_songs(
                            query_song.strip(),
                            k=k,
                            rerank_factor=rerank_factor,
                        )
                        t2 = time.time()
                        increment_run_count()
                        st.text(f"Generated in {t2 - t1:.2f}s")
                    except Exception as e:
                        st.error(f"Something went wrong while searching: {e}")
                        results = []

                if not results:
                    st.warning("No results found. Try a different description.")
                else:
                    st.subheader(f"Top {len(results)} results")
                    for i, song in enumerate(results, start=1):
                        # st.container doesn't have border param in stable streamlit; keep simple
                        cols = st.columns([1, 5])

                        with cols[0]:
                            st.markdown(f"**#{i}**")
                            if show_debug:
                                st.caption(f"ID: `{song.get('track_id', 'N/A')}`\nScore: {song.get('score', 0.0):.3f}")

                        with cols[1]:
                            st.markdown(f"**{song.get('name', 'Unknown title')}**")
                            st.markdown(song.get("artist", "Unknown artist"))
                            if show_debug and "score" in song:
                                st.caption(f"Relevance: {song['score']:.3f}")

# End of combined app

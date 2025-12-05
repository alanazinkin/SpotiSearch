import time
import streamlit as st
import torch

from song_mood_final.engine.compute_embeddings import load_embeds
from song_mood_final.engine.search_engine import SearchEngine
from song_mood_final.config.config import tok, device, base_text_model, load_config, save_config
from song_mood_final.models.text_to_features_model import TextToSpotifyFeatures
from song_mood_final.models.train_model import load_dataframe

st.set_page_config(page_title="Spotify Vibe Search", page_icon="🎧", layout="centered")

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
    state = torch.load("song_mood_final/models/spotify_model_weights.pth", map_location=device)
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

engine = get_engine()


def increment_run_count():
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


st.title("🎧 Spotify Vibe Search")
st.write("Describe a vibe and get Spotify songs that match it.")

query = st.text_input(
    "Describe the vibe:",
    placeholder="e.g., upbeat summer roadtrip pop with female vocals",
)
k = st.slider("Number of songs (k)", min_value=1, max_value=50, value=10, step=1)
go = st.button("Search")

if go and query.strip():
    with st.spinner("Searching…"):
        try:
            t1 = time.time()
            results = engine.search_songs(
                query.strip(),
                k=k,
                rerank_factor=rerank_factor,
            )
            t2 = time.time()
            increment_run_count()
            print(f'Time to Generate top {k} songs with re-rank factor of {rerank_factor}: {t2 - t1} seconds')
        except Exception as e:
            st.error(f"Something went wrong while searching: {e}")
            results = []

    if not results:
        st.warning("No results found. Try a different description.")
    else:
        st.subheader(f"Top {len(results)} results")
        for i, song in enumerate(results, start=1):
            with st.container(border=True):
                cols = st.columns([1, 5])

                with cols[0]:
                    st.markdown(f"**#{i}**")
                    if show_debug:
                        st.caption(f"ID: `{song.get('track_id', 'N/A')}`\nScore: {song.get('score', 'N/A'):.3f}")

                with cols[1]:
                    st.markdown(f"**{song.get('name', 'Unknown title')}**")
                    st.markdown(song.get("artist", "Unknown artist"))
                    if show_debug and "score" in song:
                        st.caption(f"Relevance: {song['score']:.3f}")

elif go and not query.strip():
    st.warning("Please enter a description of the vibe before searching.")

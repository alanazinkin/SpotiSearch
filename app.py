import time
import streamlit as st
import torch
from engine.compute_embeddings import load_embeds, save_new_embeddings, EMBEDDINGS_OUTPUT_PATH
from engine.search_engine import SearchEngine
from config.config import tok, device, base_text_model, load_config, save_config
from models.text_to_features_model import TextToSpotifyFeatures
from models.train_model import load_dataframe

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
def get_engine(recompute_flag=False):
    # 1. Load data (df + song_embeds)
    df = load_dataframe(filePath="data/spotify_all_songs_with_review_cols_updated.csv")
    # feature_cols, targets, texts, song_text_to_embeds = update_df(df)

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
    state = torch.load("models/spotify_model_weights.pth", map_location=device)
    spot_model.load_state_dict(state)
    spot_model.eval()

    if recompute_flag:
        with st.spinner("Embedding threshold reached. Recomputing and saving embeddings..."):
            save_new_embeddings(spot_model, df, tok, device, EMBEDDINGS_OUTPUT_PATH)

    # 4. Build SearchEngine
    song_embeds_tensor = load_embeds('data/song_embeddings.pkl', device)

    if song_embeds_tensor is None:
        raise FileNotFoundError(f"Required embeddings file missing at {'data/song_embeddings.pkl'}.")

    search_engine = SearchEngine(
        spot_model=spot_model,
        df=df,
        song_embeds=song_embeds_tensor,
        tok=tok,
        device=device,
        text_embed_model=base_text_model,
    )
    return search_engine


def initialize_app_engine():
    config = load_config()
    current_run_count = config.get('run_count', 0)
    max_runs = config.get('max_runs_before_recompute', 10)

    RECOMPUTE_EMBEDS_NEEDED = (current_run_count >= max_runs) or (current_run_count == 0)

    recompute_flag = RECOMPUTE_EMBEDS_NEEDED

    if recompute_flag:
        st.header(f"Embedding Update Cycle Triggered! ⏳ (Count: {current_run_count})")
        engine_instance = get_engine(recompute_flag=True)
        config['run_count'] = 1
        save_config(config)
        st.success(f"New embeddings loaded. Next update in {max_runs} runs.")

    else:
        engine_instance = get_engine(recompute_flag=False)

    return engine_instance

engine = initialize_app_engine()


def increment_run_count():
    config = load_config()
    current_run_count = config.get('run_count', 0)
    max_runs = config.get('max_runs_before_recompute', 10)

    new_run_count = current_run_count + 1

    config['run_count'] = new_run_count
    save_config(config)
    display_run = new_run_count if new_run_count <= max_runs else max_runs
    st.sidebar.markdown(f"**Run Count:** {display_run} / {max_runs}")


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

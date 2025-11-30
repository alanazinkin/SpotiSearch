import streamlit as st
import torch

from search_engine import SearchEngine
from config import tok, device, base_text_model
from models.text_to_features_model import TextToSpotifyFeatures
from models.train_model import load_data

st.set_page_config(page_title="Spotify Vibe Search", page_icon="🎧", layout="centered")

# ---------- Sidebar controls ----------
st.sidebar.header("Search Settings")
rerank_factor = st.sidebar.slider(
    "Rerank factor (candidate pool size)",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
    help=(
        "First retrieve k * rerank_factor candidates with the Spotify-feature model, "
        "then rerank them using Gemini-generated descriptions (for k < 30)."
    ),
)
show_debug = st.sidebar.checkbox("Show debug info (track IDs & raw scores)", value=False)

# ---------- Engine ----------

@st.cache_resource
def get_engine():
    # 1. Load data (df + song_embeds)
    df, feature_cols, targets, texts, song_embeds = load_data()

    # 2. Recreate model architecture
    out_dim = len(feature_cols)
    spot_model = TextToSpotifyFeatures(base_text_model, out_dim=out_dim).to(device)

    # 3. Load trained weights
    state = torch.load("models/spotify_model_weights.pth", map_location=device)
    spot_model.load_state_dict(state)
    spot_model.eval()

    # 4. Build SearchEngine
    search_engine = SearchEngine(
        spot_model=spot_model,
        df=df,
        song_embeds=song_embeds,
        tok=tok,
        device=device,
        text_embed_model=base_text_model,
    )
    return search_engine


engine = get_engine()

# ---------- Main page ----------
st.title("🎧 Spotify Vibe Search")
st.write("Describe a vibe and get Spotify songs that match it.")

query = st.text_input(
    "Describe the vibe:",
    placeholder="e.g., upbeat summer roadtrip pop with female vocals",
)
k = st.slider("Number of songs (k)", min_value=5, max_value=50, value=10, step=1)
go = st.button("Search")

if go and query.strip():
    with st.spinner("Searching…"):
        try:
            results = engine.search_songs(
                query.strip(),
                k=k,
                rerank_factor=rerank_factor,
            )
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

                # Rank / debug
                with cols[0]:
                    st.markdown(f"**#{i}**")
                    if show_debug:
                        st.caption(f"ID: `{song.get('track_id', 'N/A')}`\nScore: {song.get('score', 'N/A'):.3f}")

                # Song info
                with cols[1]:
                    st.markdown(f"**{song.get('name', 'Unknown title')}**")
                    st.markdown(song.get("artist", "Unknown artist"))
                    if show_debug and "score" in song:
                        st.caption(f"Relevance: {song['score']:.3f}")

elif go and not query.strip():
    st.warning("Please enter a description of the vibe before searching.")

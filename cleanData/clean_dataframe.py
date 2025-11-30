from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F


def clean_dataframe(df):
    # Define the numerical Spotify features
    feature_cols = [
        "energy", "danceability", "valence", "tempo",
        "loudness", "acousticness", "liveness",
        "speechiness", "instrumentalness",
        "time_signature", "track_popularity", "mode", "key"
    ]

    # --- NaN Handling for feature_cols ---
    # Check for NaNs and impute with mean if any are found
    if df[feature_cols].isnull().any().any():
        print("NaN values found in feature_cols. Imputing with mean...")
        for col in feature_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        print("NaN values handled.")
    else:
        print("No NaN values found in feature_cols.")

    print(f"Initial DataFrame shape: {df.shape}")

    # --- Duplicate Removal ---
    # If 'track_id' column exists, use it for duplicate removal
    if 'track_id' in df.columns:
        initial_unique_tracks = df['track_id'].nunique()
        print(f"Number of unique track_ids before dropping: {initial_unique_tracks}")
        df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
        print("Duplicates removed based on 'track_id'.")
    else:
        # If 'track_id' is not available, use 'track_name' and 'track_artist'
        print("Warning: 'track_id' column not found. Removing duplicates based on 'track_name' and 'track_artist'.")
        initial_unique_tracks = df.groupby(['track_name', 'track_artist']).ngroups
        print(f"Number of unique track_name/track_artist combinations before dropping: {initial_unique_tracks}")
        df.drop_duplicates(subset=['track_name', 'track_artist'], keep='first', inplace=True)
        print("Duplicates removed based on 'track_name' and 'track_artist'.")

    print(f"DataFrame shape after removing duplicates: {df.shape}")

    # Extract Spotify features for scaling and training targets from the de-duplicated DataFrame
    X = df[feature_cols].values.astype("float32")

    # Scale features to mean 0 / std 1 so dimensions are comparable (re-fit on cleaned data)
    scaler = StandardScaler()  # Re-initialize scaler to fit on the cleaned data
    X_scaled = scaler.fit_transform(X)

    # Turn into a normalized torch tensor -> this is your SONG EMBEDDING MATRIX for initial search
    song_embeds = torch.tensor(X_scaled)
    song_embeds = F.normalize(song_embeds, dim=-1)  # (num_songs, d)

    # Prepare the comprehensive text column for model training from the de-duplicated DataFrame
    df["text"] = (
            df["track_name"].fillna("") + " by " +
            df["track_artist"].fillna("") + ". " +
            df["playlist_genre"].fillna("") + " " +
            df["playlist_subgenre"].fillna("") + " " +
            df["playlist_name"].fillna("") + ". " +
            df["small_text"].fillna("") + " " +  # Add small_text
            df["review"].fillna("")  # Add review
    )

    # Clean up extra spaces that might result from concatenating with empty strings
    df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True).str.strip()

    texts = df["text"].tolist()
    targets = torch.tensor(X_scaled, dtype=torch.float32)  # target = scaled Spotify features

    print("\nSpotify features (X_scaled) shape:", X_scaled.shape)
    print("Song embeddings (song_embeds) shape:", song_embeds.shape)
    print("Number of texts for training:", len(texts))
    print("Targets shape:", targets.shape)
    print("First few combined texts:")
    for i in range(5):
        print(f"  - {texts[i][:150]}...")

    return feature_cols, targets, texts, song_embeds



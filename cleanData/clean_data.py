import pandas as pd

# === 1. File paths ===
songs_csv = "../data/spotifyData/high_popularity_spotify_data.csv"
low_pop_csv = "../data/spotifyData/low_popularity_spotify_data.csv"
reviews_csv = "../data/spotifyData/reviews.csv"

# Base path to save the new files
output_path = "../data/spotifyData/"

# === 2. Load datasets ===
df_high_pop_songs = pd.read_csv(songs_csv)
df_low_pop_songs = pd.read_csv(low_pop_csv)
df_reviews = pd.read_csv(reviews_csv)

# === 3. Normalize join keys & UNION ===
df_high_pop_songs["track_album_name"] = df_high_pop_songs["track_album_name"].astype(str).str.strip().str.lower()
df_low_pop_songs["track_album_name"] = df_low_pop_songs["track_album_name"].astype(str).str.strip().str.lower()
df_reviews["album"] = df_reviews["album"].astype(str).str.strip().str.lower()

# Union: df_all_songs
df_all_songs = pd.concat([df_high_pop_songs, df_low_pop_songs], ignore_index=True)

# === 4. LEFT JOIN (creates the comprehensive dataset) ===
# Perform the LEFT JOIN to keep ALL songs and add review data where available.
df_all_songs_merged = df_all_songs.merge(
    df_reviews,
    how="left",
    left_on="track_album_name",
    right_on="album"
)

# === 5. Prepare Columns & Create Final Datasets ===

# Define the final columns you want to keep
keep_cols = [
    "track_id","energy", "tempo", "danceability", "playlist_genre", "loudness",
    "liveness", "valence", "track_artist", "time_signature", "speechiness",
    "track_popularity", "track_album_name", "playlist_name", "track_name",
    "instrumentalness", "mode", "key", "acousticness", "playlist_subgenre",
    "rating", "small_text", "review",
]

# --- Dataset 1: All Songs (with Nulls) ---
# Filter the comprehensive dataset to only contain the specified columns
# Rows without a review will have NaN in 'rating' and 'text_review'.
# Check for and drop the redundant 'album' column if it exists and is not in keep_cols
if "album" in df_all_songs_merged.columns:
    df_all_songs_merged = df_all_songs_merged.drop(columns=["album"])

# Select the final columns (handle potential missing columns if the merge failed to provide them all)
missing_cols = [col for col in keep_cols if col not in df_all_songs_merged.columns]
if missing_cols:
    print(f"Warning: Columns not found after merge/rename: {missing_cols}. They will be skipped.")
    # Use only the columns that actually exist
    available_cols = [col for col in keep_cols if col in df_all_songs_merged.columns]
    df_all_songs_with_reviews = df_all_songs_merged[available_cols]
else:
    df_all_songs_with_reviews = df_all_songs_merged[keep_cols]

NEW_COLUMN_NAME = "gemini_review"
if NEW_COLUMN_NAME not in df_all_songs_with_reviews.columns:
    df_all_songs_with_reviews[NEW_COLUMN_NAME] = pd.NA
    df_all_songs_with_reviews[NEW_COLUMN_NAME] = df_all_songs_with_reviews[NEW_COLUMN_NAME].astype(object)
keep_cols.append(NEW_COLUMN_NAME)

# --- Dataset 2: Reviewed Songs Only (Filtered) ---
# Filter the comprehensive dataset to keep only rows where 'text_review' is not NaN
df_reviewed = df_all_songs_with_reviews[df_all_songs_with_reviews["review"].notna()]


# === 6. Save the DataFrames to Drive ===
reviewed_file = output_path + "spotify_songs_reviewed_only.csv"
all_songs_file = output_path + "spotify_all_songs_with_review_cols.csv"

df_reviewed.to_csv(reviewed_file, index=False)
df_all_songs_with_reviews.to_csv(all_songs_file, index=False)


# === 7. Print Summary ===
print(f"--- Data Preparation Complete! ---")
print(f"Dataset 1 (df_reviewed) created: {reviewed_file}")
print(f"  - Rows (Songs with Reviews): {len(df_reviewed)}")
print("-" * 40)
print(f"Dataset 2 (df_all_songs_with_reviews) created: {all_songs_file}")
print(f"  - Rows (All Songs): {len(df_all_songs_with_reviews)}")
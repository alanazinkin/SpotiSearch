import pandas as pd
import unicodedata
import re

FILE_MYALBUMS = "building_embedding_data/myalbums.csv"
FILE_REVIEWS = "data_cleaning_data/reviews_with_moods.csv"
OUTPUT_FILE = "building_embedding_data/albums_in_both.csv"

def normalize(s):
    """Normalize for case-insensitive album matching."""
    if pd.isna(s):
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)   # replace punctuation with space
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Load datasets
df_my = pd.read_csv(FILE_MYALBUMS, dtype=str, keep_default_na=False)
df_rev = pd.read_csv(FILE_REVIEWS, dtype=str, keep_default_na=False)

# Add normalized album columns
df_my["album_norm"] = df_my["Album Name"].apply(normalize)
df_rev["album_norm"] = df_rev["title"].apply(normalize)

# Inner join on normalized album
# Prepare unique album normals from `myalbums` to avoid duplicate matches
my_albums_unique = df_my[["album_norm"]].drop_duplicates()

# Inner join on normalized album (this brings rows from reviews that match
# any album in `myalbums`). Use the unique set so we don't multiply matches
# because of duplicate entries in `myalbums`.
merged = df_rev.merge(
    my_albums_unique,
    on="album_norm",
    how="inner"
)

# If multiple reviews exist for the same album, collapse to a single row
# per normalized album to avoid duplicates in the output. Keep the first
# matched review for that album.
before = len(merged)
merged = merged.drop_duplicates(subset=["album_norm"], keep="first")
after = len(merged)

# Drop helper normalization column before writing output
merged = merged.drop(columns=["album_norm"])
print(f"Matched rows before dedupe: {before}; after dedupe: {after}")

# Write output: only rows & columns from reviews_with_moods.csv
merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"Matched rows: {len(merged)}")
print(f"Saved to: {OUTPUT_FILE}")

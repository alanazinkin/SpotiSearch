import random
import csv
import json
import pandas as pd
import re

with open("data_cleaning_data/music_descriptors_3000.csv", "r") as f:
    text = f.read()

items = [x.strip() for x in text.split(",")]
genres_descriptors_set = set(items)

with open("genres.json", "r") as f:
    data = json.load(f)

# Build a set from the "name" field
name_set = (item["name"] for item in data)
genres_descriptors_set.update(name_set)
genres_descriptors_set = {x.lower() for x in genres_descriptors_set}

# Load adjectives from Adjectives.csv into a separate set
adjectives_set = set()
try:
    adj_df = pd.read_csv("data_cleaning_data/Adjectives.csv")
    # Prefer a column named 'word' or 'Word' if present
    cols_lower = [c.lower() for c in adj_df.columns]
    if "word" in cols_lower:
        col = adj_df.columns[cols_lower.index("word")]
        adj_series = adj_df[col].astype(str)
    else:
        # Fallback: pick the first text-like column, or stack all values
        text_cols = [c for c in adj_df.columns if adj_df[c].dtype == object]
        if text_cols:
            adj_series = adj_df[text_cols[0]].astype(str)
        else:
            adj_series = adj_df.astype(str).stack()

    adj_series = adj_series.str.strip().str.lower()
    # sanitize entries: remove stray punctuation
    adj_series = adj_series.str.replace(r"[^a-z0-9\-\s]", "", regex=True).str.strip()
    adjectives_set.update(w for w in adj_series if w)
except FileNotFoundError:
    # It's fine if the file isn't present; just continue
    pass
except Exception as e:
    print(f"Warning: could not load Adjectives.csv: {e}")

# print(len(genres_descriptors_set))
df = pd.read_csv("data_cleaning_data/unique_albums.csv")

def extract_mood_words_skip_negation(text, word_set):
    """Extract words from word_set, skipping any word that follows 'not'."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # split
    words = text.split()
    # filter: skip words that follow 'not'
    found = []
    skip_next = False
    for w in words:
        if skip_next:
            skip_next = False
            continue
        if w == "not":
            skip_next = True
            continue
        if w in word_set:
            found.append(w)
    return " ".join(set(found))  

df["reviews_adjectives"] = df["review"].apply(lambda x: extract_mood_words_skip_negation(x, adjectives_set))
df["query"] = df["review"].apply(lambda x: extract_mood_words_skip_negation(x, genres_descriptors_set))

# Prepend adjectives to small_text column
df["small_text"] = df["reviews_adjectives"] + " " + df["small_text"].astype(str)

# Add sparseness flag (based on mood_words, i.e., genres/descriptors)
df["is_sparse_mood"] = df["query"].str.split().str.len() <= 1

# Drop rows where mood is sparse (True)
before_count = len(df)
df = df[~df["is_sparse_mood"]]
after_count = len(df)
print(f"Dropped {before_count - after_count} sparse-mood rows; {after_count} remain.")

# Drop the temporary reviews_adjectives column
df = df.drop(columns=["reviews_adjectives"])
df = df.drop(columns=["label"])
df = df.drop(columns=["is_sparse_mood"])
df = df.drop(columns=["review"])

# Rename small_text to review
df = df.rename(columns={"small_text": "review"})
df = df.rename(columns={"album": "title"})


# Create unique album ID based on artist + album combination
df["album_id"] = pd.factorize(df["artist"] + " - " + df["title"])[0]
df.to_csv("data_cleaning_data/reviews_with_moods.csv", index=False)

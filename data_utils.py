import pandas as pd

def generate_text_for_embedding(df: pd.DataFrame) -> list[str]:
    """
    Creates the concatenated text string for each song in the DataFrame.
    """
    text_series = (
        df["track_name"].fillna("") + " by " +
        df["track_artist"].fillna("") + ". " +
        df["playlist_genre"].fillna("") + " " +
        df["playlist_subgenre"].fillna("") + " " +
        df["playlist_name"].fillna("") + ". " +
        df["gemini_review"].fillna("") + " " +
        df["small_text"].fillna("")
    )
    return text_series.tolist()
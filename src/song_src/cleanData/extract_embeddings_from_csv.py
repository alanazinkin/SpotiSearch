import pickle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.song_src.cleanData.clean_dataframe import clean_dataframe
from src.song_src.engine.search_engine import save_df

SPOTIFY_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature',
]


def create_song_feature_vectors(
        input_csv_path: str,
        output_pkl_path: str,
        scaler_path: str = 'spotify_feature_scaler.pkl'
) -> None:
    print(f"Loading data from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {input_csv_path}")
        return

    # 1. Select the numerical features to scale
    df, feature_cols, targets, texts, song_embeds = clean_dataframe(df)
    save_df(df=df, path=input_csv_path)
    df_cols_map = {col.lower(): col for col in df.columns}
    features_to_scale = []
    for feature in SPOTIFY_FEATURES:
        lower_feature = feature.lower()
        if lower_feature in df_cols_map:
            features_to_scale.append(df_cols_map[lower_feature])

    if len(features_to_scale) < len(SPOTIFY_FEATURES):
        print(f"Warning: Found only {len(features_to_scale)} of {len(SPOTIFY_FEATURES)} required features.")
    elif len(features_to_scale) > len(SPOTIFY_FEATURES):
        print(f"Embedding dimension is incorrect. There are {len(features_to_scale)} features instead of {len(SPOTIFY_FEATURES)} required features. "
              f"Ensure your dataframe contains 12 required columns")
        return
    X = df[features_to_scale].values

    # 2. Initialize and fit the scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Convert the scaled array to the required dictionary format
    # Keys will be the DataFrame index (which should correspond to your Song ID/Index)
    # Values will be the scaled feature vector (converted to list for saving)
    song_feature_dict = {
        idx: X_scaled[i].tolist()
        for i, idx in enumerate(df.index)
    }

    # 4. Save the Feature Dictionary to PKL using the pickle module
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(song_feature_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Normalized feature vectors saved to: {output_pkl_path} (Count: {len(song_feature_dict)})")

    # 5. Save the fitted Scaler object (Remains the same, using joblib)
    joblib.dump(scaler, scaler_path)
    print(f"MinMaxScaler saved to: {scaler_path}")

    print("\n--- Process Complete ---")


if __name__ == "__main__":
    create_song_feature_vectors(
        input_csv_path="../../../data/song_data/My_Liked_Songs.csv",
        output_pkl_path="../../../data/song_data/my_song_features_normalized.pkl"
    )
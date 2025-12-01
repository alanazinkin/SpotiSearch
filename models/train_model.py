from torch.utils.data import DataLoader
from models.text_to_features_model import TextToSpotifyFeatures, SongTextDataset
import pandas as pd
import torch
from torch import nn
from cleanData.clean_dataframe import clean_dataframe
from config import device, base_text_model


def load_data(filePath= "../data/spotifyData/spotify_all_songs_with_review_cols.csv"):
    df = pd.read_csv(filePath)
    print(df.head())
    feature_cols, targets, texts, song_embeds = clean_dataframe(df)
    return df, feature_cols, targets, texts, song_embeds


def train_model(model, loader, loss_fn, opt, n_epochs: int = 10, save_path: str | None = None):
    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            pred = model(input_ids, attn)
            target = batch["target"].to(device)  # (batch, d)

            loss = loss_fn(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * input_ids.size(0)

        print(f"epoch {epoch + 1}: {total / len(loader.dataset):.4f}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model parameters saved to {save_path}")


# Only run training if this file is executed directly:
#   python -m models.train_model
if __name__ == "__main__":
    df, feature_cols, targets, texts, song_embeds = load_data("../data/spotifyData/spotify_all_songs_with_review_cols_updated.csv")

    dataset = SongTextDataset(texts, targets)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    spot_model = TextToSpotifyFeatures(base_text_model, out_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(spot_model.parameters(), lr=1e-4)
    loss_function = nn.MSELoss()

    print("Dataset and DataLoader re-instantiated with updated data.")
    print(f"Model initialized with output dimension: {len(feature_cols)}")

    train_model(
        spot_model,
        data_loader,
        loss_function,
        optimizer,
        n_epochs=10,
        save_path="spotify_model_weights.pth",
    )
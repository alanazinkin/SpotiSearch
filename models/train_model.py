from torch.utils.data import DataLoader
from models.text_to_features_model import TextToSpotifyFeatures, SongTextDataset
import pandas as pd
import torch
from torch import nn
from cleanData.clean_dataframe import clean_dataframe
from config import device, base_text_model
from sklearn.model_selection import train_test_split


def load_dataframe(filePath= "../data/spotifyData/spotify_all_songs_with_review_cols.csv"):
    df = pd.read_csv(filePath)
    print(df.head())
    return df

def update_df(dataframe: pd.DataFrame):
    feature_cols, targets, texts, song_text_to_embed = clean_dataframe(dataframe)
    return feature_cols, targets, texts, song_text_to_embed

def evaluate_model(model, loader, loss_fn):
    """Calculates the loss on the provided dataset loader (typically the test set)."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            target = batch["target"].to(device)

            pred = model(input_ids, attn)
            loss = loss_fn(pred, target)
            total_loss += loss.item() * input_ids.size(0)

    # Return the average loss (Mean Squared Error)
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, test_loader, loss_fn, opt, n_epochs: int = 10, save_path: str | None = None):
    for epoch in range(n_epochs):
        # --- Training Step ---
        model.train()
        train_total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pred = model(input_ids, attn)
            target = batch["target"].to(device)

            loss = loss_fn(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_total_loss += loss.item() * input_ids.size(0)

        test_mse = evaluate_model(model, test_loader, loss_fn)

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Train MSE: {train_total_loss / len(train_loader.dataset):.4f} | "
            f"Test MSE: {test_mse:.4f}"
        )

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model parameters saved to {save_path}")


if __name__ == "__main__":
    df = load_dataframe("../data/spotify_all_songs_with_review_cols_updated.csv")
    feature_cols, targets, texts, song_text_to_embed = update_df(df)

    train_texts, test_texts, train_targets, test_targets = train_test_split(
        texts,
        targets,
        test_size=0.2,
        random_state=42
    )

    # 1. Create Training Dataset and DataLoader
    train_dataset = SongTextDataset(train_texts, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = SongTextDataset(test_texts, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    spot_model = TextToSpotifyFeatures(base_text_model, out_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(spot_model.parameters(), lr=1e-4)
    loss_function = nn.MSELoss()

    print(f"Total Samples: {len(df)}")
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Test Samples: {len(test_dataset)}")
    print(f"Model initialized with output dimension: {len(feature_cols)}")

    train_model(
        spot_model,
        train_loader,
        test_loader,
        loss_function,
        optimizer,
        n_epochs=5,
        save_path="spotify_model_weights.pth",
    )
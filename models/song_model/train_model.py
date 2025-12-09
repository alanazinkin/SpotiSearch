from torch.utils.data import DataLoader
from models.song_model.text_to_features_model import TextToSpotifyFeatures, SongTextDataset
import pandas as pd
import torch
from torch import nn
from src.song_src.cleanData.clean_dataframe import clean_dataframe
from src.song_src.config.config import load_config, save_config
from models.song_model.sentence_embedding_model import device, base_text_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def load_dataframe(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    return df

def update_df(dataframe: pd.DataFrame):
    return clean_dataframe(dataframe)

def reset_run_counter():
    try:
        config = load_config()
        if config.get('run_count', 0) != 0:
            config['run_count'] = 0
            save_config(config)
            print("Successfully RESET run_count to 0 in config.json.")
        else:
            print("Run counter was already 0. No reset necessary.")

    except Exception as e:
        print(f"Warning: Could not reset run counter. Error: {e}")

def evaluate_model(model, loader, loss_fn):
    """Calculates the loss on the provided dataset loader (typically the validation set)."""
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


train_losses = []
val_losses = []

'''Leveraged Gemini LLM to assist with patience'''
def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    opt,
    n_epochs: int = 10,
    patience: int = 5,
    save_path: str | None = None,
):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

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


        current_val_loss = evaluate_model(model, val_loader, loss_fn)
        avg_train_loss = train_total_loss / len(train_loader.dataset)

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Train MSE: {avg_train_loss:.4f} | "
            f"Validation MSE: {current_val_loss:.4f}"
        )
        train_losses.append(avg_train_loss)
        val_losses.append(current_val_loss)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"    ⭐ Validation loss improved. Saving current model state.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs (Patience {patience} exceeded).")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print("Loaded best model weights.")
            break

    # --- Final Save ---
    # Save the *best* model state (the one with the lowest validation loss)
    if save_path is not None and best_model_state is not None:
        # If the loop completed normally, or broke early, save the best state found.
        torch.save(best_model_state, save_path)
        print(f"\nFinal best model parameters saved to {save_path} (Best Val MSE: {best_val_loss:.4f})")



# Helper function to visualize performance during training, courtsey of HW4
def plot_training_curves(train_losses, val_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.plot(val_losses)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_dataframe("/content/drive/MyDrive/spotifySongData/spotify_all_songs_with_review_cols_updated.csv")
    df, feature_cols, targets, texts, song_text_to_embed = update_df(df)

    train_texts, temp_texts, train_targets, temp_targets = train_test_split(
    texts, targets, test_size=0.2, random_state=42
    )

    # 2. Split temporary data into validation (50% of temp) and test (50% of temp)
    # Final split of 80% Train, 10% Validation, 10% Test
    val_texts, test_texts, val_targets, test_targets = train_test_split(
        temp_texts, temp_targets, test_size=0.5, random_state=42
    )

    # Create the Validation Loader
    val_dataset = SongTextDataset(val_texts, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
        val_loader,
        loss_function,
        optimizer,
        n_epochs=10,
        save_path="spotify_model_weights.pth",
        patience=3
    )
    plot_training_curves(train_losses, val_losses)
    final_test_mse = evaluate_model(spot_model, test_loader, loss_function)
    print(f"\n--- Final Test Evaluation ---")
    print(f"Final Test MSE: {final_test_mse:.4f}")
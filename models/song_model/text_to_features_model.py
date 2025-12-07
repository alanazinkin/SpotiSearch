from torch import nn
from torch.utils.data import Dataset
from models.song_model.sentence_embedding_model import tok

class SongTextDataset(Dataset):
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": self.targets[i],
        }

class TextToSpotifyFeatures(nn.Module):
    def __init__(self, encoder, out_dim):
        super().__init__()
        self.encoder = encoder
        hidden_dim = encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = (out.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
        x = x / attention_mask.sum(1, keepdim=True)
        return self.head(x)
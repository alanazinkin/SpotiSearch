import torch
import json
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(text_model_name)
base_text_model = AutoModel.from_pretrained(text_model_name).to(device)

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'run_count': 0, 'max_runs_before_recompute': 10}

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
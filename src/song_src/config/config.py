import json

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'run_count': 0, 'max_runs_before_recompute': 10}

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
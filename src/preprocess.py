import os
import json
from torchaudio import load

def load_data(data_dir):
    data_pairs = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            json_path = os.path.join(data_dir, file)
            mp3_path = json_path.replace(".json", ".mp3")
            
            if os.path.exists(mp3_path):
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                data_pairs.append((metadata["keyword"], mp3_path))
    return data_pairs

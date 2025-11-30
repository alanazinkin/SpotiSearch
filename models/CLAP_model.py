import torch
from transformers import ClapModel, AutoProcessor

class ClapAudioEncoder:
    def __init__(self, model_name: str = "laion/clap-htsat-unfused", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def encode_audio_file(self, audio_path: str) -> torch.Tensor:
        """
        Returns a single embedding tensor of shape (d,)
        """
        audio, sr = self.processor.audio_read(audio_path)
        inputs = self.processor(audios=audio, sampling_rate=sr, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model.get_audio_features(**inputs)  # shape: (1, d)

        return out.squeeze(0).cpu()  # (d,)

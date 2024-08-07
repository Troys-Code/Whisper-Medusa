import torch
import torchaudio
from whisper_medusa import WhisperMedusaModel
from transformers import WhisperProcessor

class SpeechToText:
    def __init__(self, model_name="aiola/whisper-medusa-v1", sampling_rate=16000, device=None):
        self.model_name = model_name
        self.sampling_rate = sampling_rate
        # self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.device = torch.device("cpu")
        self.model = WhisperMedusaModel.from_pretrained(self.model_name)
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)

    def transcribe(self, path_to_audio):
        # Load audio file
        input_speech, sr = torchaudio.load(path_to_audio)
        print("Loaded audio shape:", input_speech.shape)
        
        if sr != self.sampling_rate:
            input_speech = torchaudio.transforms.Resample(sr, self.sampling_rate)(input_speech)
            print("Resampled audio shape:", input_speech.shape)

        # Convert to mono if needed
        if input_speech.shape[0] == 2:
            input_speech = torch.mean(input_speech, dim=0, keepdim=True)

        # Prepare input features
        try:
            input_features = self.processor(input_speech.squeeze(), return_tensors="pt", sampling_rate=self.sampling_rate).input_features
            print("Input features shape:", input_features.shape)
        except Exception as e:
            print("Error processing input features:", e)
            raise

        input_features = input_features.to(self.device)

        # Generate transcription
        model_output = self.model.generate(
            input_features,
            language="en",
        )
        predict_ids = model_output[0]
        pred = self.processor.decode(predict_ids, skip_special_tokens=True)
        return pred

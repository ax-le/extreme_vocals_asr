import librosa
import numpy as np
from evaluate import load

import model_loaders

song_path = TODO
# song = librosa.load(song_path, sr=16000)[0] # Probably uselesss

ground_truth_lyrics = TODO

# Source Separation ?
source_separated = TODO

# Apply an ASR model
asr_model = model_loaders.load_model("Whisper") # Accepts "Whisper", "Canary", "Wav2vec"
predicted_lyrics = asr_model.transcribe(song_path)

wer = load("wer")
wer_score = wer.compute(predictions=predicted_lyrics, references=ground_truth_lyrics)
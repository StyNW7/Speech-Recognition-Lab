import sys
import torch
import torchaudio
import soundfile as sf

# torch       -> tensor & neural network
# torchaudio  -> model pretrained wav2vec 2.0 + resample
# soundfile   -> membaca file .wav (torchaudio.load butuh torchcodec sejak versi 2.9)

# Cara pakai:
#   python wav2vec.py                  -> transkripsi speech.wav
#   python wav2vec.py rekaman.wav      -> transkripsi file lain (bahasa Inggris)
AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else "./speech.wav"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LOAD MODEL PRETRAINED
# WAV2VEC2_ASR_BASE_960H = wav2vec 2.0 base, sudah di-fine-tune 960 jam LibriSpeech (Inggris)
# pertama kali dijalankan, bobot model (+-360 MB) akan di-download otomatis
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
model.eval()
labels = bundle.get_labels()   # ('-', '|', 'E', 'T', ...) -> '-' = blank, '|' = spasi

# 2. LOAD AUDIO
audio, sample_rate = sf.read(AUDIO_PATH, dtype="float32")
waveform = torch.from_numpy(audio)
if waveform.ndim == 2:                 # stereo -> mono
    waveform = waveform.mean(dim=1)
waveform = waveform.unsqueeze(0)       # (samples,) -> (1, samples)

# 3. RESAMPLE -> model hanya mengerti audio 16 kHz
if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
waveform = waveform.to(device)

with torch.inference_mode():
    # 4. EKSTRAKSI FITUR -> output 12 layer Transformer, shape (1, frame, 768), 1 frame = 20 ms
    features, _ = model.extract_features(waveform)

    # 5. EMISSION -> skor tiap label per frame, shape (1, frame, 29)
    emission, _ = model(waveform)


# 6. CTC GREEDY DECODER
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)           # label terbaik per frame
        indices = torch.unique_consecutive(indices)        # gabungkan yang berulang
        indices = [i for i in indices if i != self.blank]  # buang blank
        text = "".join(self.labels[i] for i in indices)
        return text.replace("|", " ").strip()              # '|' -> spasi


decoder = GreedyCTCDecoder(labels)
transcript = decoder(emission[0].cpu())

print("File       :", AUDIO_PATH)
print("Durasi     :", round(waveform.shape[1] / bundle.sample_rate, 2), "detik")
print("Fitur      :", len(features), "layer x", tuple(features[0].shape))
print("Emission   :", tuple(emission.shape))
print("Transkrip  :", transcript)

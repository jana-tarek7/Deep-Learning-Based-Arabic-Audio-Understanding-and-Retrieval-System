import torch
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# WHISPER
# =========================
whisper_model_name = "ayoubkirouane/whisper-small-ar"

whisper_processor = AutoProcessor.from_pretrained(whisper_model_name)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_name).to(device)


# =========================
# WAV2VEC2
# =========================
wav2vec_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
wav2vec_model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_name).to(device)


# =========================
# AUDIO LOADER (from utils)
# =========================
def load_audio(path, sr=16000):
    import torchaudio

    wav, sample_rate = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sample_rate != sr:
        wav = torchaudio.functional.resample(wav, sample_rate, sr)

    return wav.squeeze(0)


# =========================
# WHISPER TRANSCRIBE
# =========================
def transcribe_whisper(path):
    audio = load_audio(path)

    inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").to(device)

    with torch.no_grad():
        ids = whisper_model.generate(inputs.input_features)

    return whisper_processor.batch_decode(ids, skip_special_tokens=True)[0]


# =========================
# WAV2VEC TRANSCRIBE
# =========================
def transcribe_wav2vec(path):
    audio = load_audio(path)

    inputs = wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        logits = wav2vec_model(inputs.input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)

    return wav2vec_processor.batch_decode(pred_ids)[0]


# =========================
# MAIN SWITCH FUNCTION
# =========================
def transcribe_audio(path, model_type="whisper"):

    if model_type == "whisper":
        return transcribe_whisper(path)

    elif model_type == "wav2vec":
        return transcribe_wav2vec(path)

    else:
        raise ValueError("model_type must be 'whisper' or 'wav2vec'")

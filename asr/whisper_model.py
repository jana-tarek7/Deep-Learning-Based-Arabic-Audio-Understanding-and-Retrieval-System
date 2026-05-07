from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ayoubkirouane/whisper-small-ar"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
model.eval()

def transcribe_whisper(audio_inputs):
    with torch.no_grad():
        ids = model.generate(
            audio_inputs.input_features,
            language="ar",
            task="transcribe"
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

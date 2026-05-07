from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
model.eval()

def transcribe_wav2vec(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]

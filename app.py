import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch

from asr.audio_utils import load_audio
from asr.whisper_model import processor as whisper_processor, transcribe_whisper
from asr.wav2vec_model import transcribe_wav2vec
from summarization.summarizer import summarize

st.set_page_config(page_title="Arabic Audio AI System")

st.title("🎙️ Arabic Audio Understanding System")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

model_choice = st.selectbox("Choose ASR Model", ["Whisper", "Wav2Vec2"])

if uploaded_file is not None:

    path = f"temp_audio.wav"
    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(path)

    audio = load_audio(path)

    st.subheader("🧠 Transcription")

    if model_choice == "Whisper":
        inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
        transcript = transcribe_whisper(inputs)
    else:
        transcript = transcribe_wav2vec(audio)

    st.write(transcript)

    st.subheader("📝 Summary")
    summary = summarize(transcript)
    st.write(summary)

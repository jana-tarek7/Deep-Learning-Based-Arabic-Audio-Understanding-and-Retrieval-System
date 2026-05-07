import streamlit as st
from src.asr import transcribe
from src.summarization import summarize

st.title("Arabic Audio Understanding System")

audio_file = st.file_uploader("Upload audio file")

if audio_file:
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    text = transcribe("temp.wav")
    summary = summarize(text)

    st.subheader("Transcript")
    st.write(text)

    st.subheader("Summary")
    st.write(summary)

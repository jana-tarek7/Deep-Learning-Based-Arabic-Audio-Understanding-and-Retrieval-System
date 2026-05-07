import streamlit as st
from src.speech_to_text import transcribe_audio
from src.summarization import summarize_text
from src.search import search_query

st.title("🎧 Audio AI Project")

audio = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if audio:
    st.audio(audio)

    st.write("🔄 Transcribing...")
    text = transcribe_audio(audio)

    st.subheader("Transcript")
    st.write(text)

    st.write("🧠 Summarizing...")
    summary = summarize_text(text)

    st.subheader("Summary")
    st.write(summary)

    query = st.text_input("Search in audio")

    if query:
        results = search_query(query)
        st.write(results)

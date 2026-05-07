from pipeline.asr import transcribe_audio
from pipeline.summarizer import summarize_text
from pipeline.search import search_context


def run_pipeline(audio_path, query):

    # 1. Speech to text
    transcript = transcribe_audio(audio_path)

    # 2. Summarization
    summary = summarize_text(transcript)

    # 3. Search
    result = search_context(query)

    return {
        "transcript": transcript,
        "summary": summary,
        "search_result": result
    }

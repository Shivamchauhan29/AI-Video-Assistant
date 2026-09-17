import os
import uuid
from dotenv import load_dotenv


load_dotenv()

def run_pipeline(source :str, language :str = "english") -> dict:
    from utils.audio_processor import process_input, acquire_video_source
    from core.transcriber import transcribe_all
    from core.summarizer import summarize, generate_title
    from core.extractor import extract_action_items, extract_key_decisions, extract_questions
    from core.rag_engine import build_rag_chain, ask_question
    from core.highlighter import generate_highlights
    from core.clipper import cut_clips

    print("starting AI Video Assistant")

    chunks = process_input(source)

    transcription = transcribe_all(chunks, language)
    transcript = transcription["text"]
    transcript_segments = transcription["segments"]
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_item = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    
    rag_chain = build_rag_chain(transcript)

    # English/Whisper path only — transcript_segments is None for Hinglish/Sarvam.
    highlights = generate_highlights(transcript_segments) if transcript_segments else []

    if highlights:
        try:
            video_path = acquire_video_source(source)
            clip_dir = os.path.join("clips", uuid.uuid4().hex)
            highlights = cut_clips(video_path, highlights, clip_dir)
        except Exception as clip_exc:
            # Clip cutting is a bonus on top of an already-successful
            # analysis — don't let it take down the whole result.
            print(f"Clip cutting failed (non-fatal): {clip_exc}")

    return {
        "title": title,
        "transcript": transcript,
        "transcript_segments": transcript_segments,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "highlights": highlights,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    if result["highlights"]:
        print("\n🎯 Highlight Candidates:")
        for clip in result["highlights"]:
            path_info = f"  -> {clip['path']}" if clip.get("path") else ""
            print(f"  [{clip['start']:.1f}s - {clip['end']:.1f}s] score={clip['score']}  {clip['reason']}{path_info}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")
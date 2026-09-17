from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

from core.pipeline_logger import log_if

def get_llm():
    # Groq is primary (fast, generous free tier); Mistral is the fallback
    # if Groq errors (rate limit, outage, etc.) via LangChain's built-in
    # runnable fallback — no manual try/except needed at call sites.
    groq_llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)
    mistral_llm = ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.3)
    return groq_llm.with_fallbacks([mistral_llm])


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

def summarize(transcript : str, logger=None) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    log_if(logger, "Summarization", f"Split transcript into {len(chunks)} chunk(s)")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_summaries.append(map_chain.invoke({"text": chunk}))
        log_if(logger, "Summarization", f"Summarized chunk {i + 1}/{len(chunks)}")

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    log_if(logger, "Summarization", "Combining chunk summaries...")
    result = combined_chain.invoke(combined)
    log_if(logger, "Summarization", "Summary complete")
    return result

def generate_title(transcipt : str, logger=None) -> str:
    llm = get_llm()

    log_if(logger, "Title", "Generating title...")

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) |
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    result = title_chain.invoke(transcipt[:2000])
    log_if(logger, "Title", "Title generated")
    return result





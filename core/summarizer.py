from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.pipeline_logger import log_if
from core.llm_provider import get_llm as _get_llm


def get_llm(logger=None):
    return _get_llm(temperature=0.3, logger=logger)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

def summarize(transcript : str, logger=None) -> str:
    llm = get_llm(logger=logger)

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
    llm = get_llm(logger=logger)

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





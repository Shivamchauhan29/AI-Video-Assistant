#Actionableitems , decision , questions 

from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os

from core.pipeline_logger import log_if


def get_llm():
    # Groq is primary; falls back to Mistral automatically on error (see summarizer.py).
    groq_llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)
    mistral_llm = ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.2)
    return groq_llm.with_fallbacks([mistral_llm])



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str, logger=None)->str:
    log_if(logger, "Extraction", "Extracting action items...")
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    result = chain.invoke(transcript)
    log_if(logger, "Extraction", "Action items extracted")
    return result


def extract_key_decisions(transcript: str, logger=None) -> str:
    log_if(logger, "Extraction", "Extracting key decisions...")
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    result = chain.invoke(transcript)
    log_if(logger, "Extraction", "Key decisions extracted")
    return result


def extract_questions(transcript: str, logger=None) -> str:
    log_if(logger, "Extraction", "Extracting open questions...")
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    result = chain.invoke(transcript)
    log_if(logger, "Extraction", "Open questions extracted")
    return result
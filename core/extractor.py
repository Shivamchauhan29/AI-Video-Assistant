#Actionableitems , decision , questions

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.pipeline_logger import log_if
from core.llm_provider import get_llm as _get_llm


def get_llm(logger=None):
    return _get_llm(temperature=0.2, logger=logger)


def build_chain(system_prompt : str, logger=None):
    llm = get_llm(logger=logger)
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
        "Format as a numbered list. If none found say 'No action items found.'",
        logger=logger,
    )

    result = chain.invoke(transcript)
    log_if(logger, "Extraction", "Action items extracted")
    return result


def extract_key_decisions(transcript: str, logger=None) -> str:
    log_if(logger, "Extraction", "Extracting key decisions...")
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'",
        logger=logger,
    )
    result = chain.invoke(transcript)
    log_if(logger, "Extraction", "Key decisions extracted")
    return result


def extract_questions(transcript: str, logger=None) -> str:
    log_if(logger, "Extraction", "Extracting open questions...")
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'",
        logger=logger,
    )
    result = chain.invoke(transcript)
    log_if(logger, "Extraction", "Open questions extracted")
    return result
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever
from core.pipeline_logger import log_if
from core.llm_provider import get_llm as _get_llm

def get_llm(logger=None):
    return _get_llm(temperature=0.3, logger=logger)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript:str, logger=None):

    log_if(logger, "RAG Engine", "Building retrieval chain...")
    vector_store = build_vector_store(transcript, logger=logger)

    retriever = get_retriever(vector_store, k = 4)

    llm = get_llm(logger=logger)

    prompt = ChatPromptTemplate.from_messages(

        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ]
    )

    #full LCEL Rag pipeline 

    rag_chain = (

        {"context" : retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
         |prompt|llm|StrOutputParser()
    )

    log_if(logger, "RAG Engine", "RAG chain ready")
    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriver = get_retriever()

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context":  retriver| RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question:str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer
# Initialize Langfuse handler
import os

from dotenv import load_dotenv
from langfuse.callback import CallbackHandler

load_dotenv()
langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


def format_docs(docs: list[Document]) -> str:
    return "\n".join([doc.page_content for doc in docs])


def main():
    docs = [
        Document(page_content="Jesse loves red but not yellow"),
        Document(page_content="Jamal loves green but not as much as he loves orange"),
    ]
    vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "以下のcontextだけに基づいて回答してください。\n=====\n{context}\n=====\n質問: {question}",
            )
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, sample=False)

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs, name="FormatDocs"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    # Add Langfuse handler as callback (classic and LCEL)
    result = chain.invoke(
        "What are everyone's favorite colors?", config={"callbacks": [langfuse_handler]}
    )
    print(result)

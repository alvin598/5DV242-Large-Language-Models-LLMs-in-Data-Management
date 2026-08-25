from abc import ABC, abstractmethod

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from data.documents import urls


class BaseRag(ABC):
    def __init__(self):
        self.urls = urls
        self.retriever = self.indexing()

        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. 

            Use the following documents to answer the question. 

            If you don't know the answer, just say that you don't know. 

            Use three sentences maximum and keep the answer concise:
            Question: {utterance} 
            Documents: {documents} 
            Answer: 
            """,
            input_variables=["utterance", "documents"],
        )

        llm = ChatOllama(
            model="llama3.1",
            temperature=0,
        )

        self.rag_chain = prompt | llm | StrOutputParser()

    @abstractmethod
    def indexing(self):
        raise NotImplementedError("Implemented in subclass")

    def retrieve_documents(self, utterance):
        return self.retriever.invoke(utterance)

    def run(self):
        while True:
            utterance = input(">")
            if utterance == "quit":
                break

            documents = self.retrieve_documents(utterance)

            print("\nRetrieved documents:")
            for number, document in enumerate(documents, start=1):
                source = document.metadata.get("source", "unknown source")
                preview = document.page_content[:200].replace("\n", " ")
                print(f"{number}. {source}")
                print(f"   {preview}...")

            response = self.rag_chain.invoke(
                {"utterance": utterance, "documents": documents}
            )
            print(f"{response}")
        
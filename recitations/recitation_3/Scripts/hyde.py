from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

from Scripts.BaseRag import BaseRag


class HyDE(BaseRag):
    def __init__(self):
        super().__init__()

        hypothetical_prompt = PromptTemplate(
            template="""Write a short hypothetical passage that could answer the question.
                Do not mention that the passage is hypothetical. Include relevant facts and terminology.
                Question: {utterance}
                Passage:""",
            input_variables=["utterance"],
        )
        llm = ChatOllama(model="llama3.1", temperature=0)
        self.hypothetical_chain = hypothetical_prompt | llm | StrOutputParser()

    def indexing(self):
        docs = [
            WebBaseLoader(
                url,
                header_template={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
                },
            ).load()
            for url in self.urls
        ]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=NomicEmbeddings(
                model="nomic-embed-text-v1.5", inference_mode="local"
            ),
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    def retrieve_documents(self, utterance):
        hypothetical_answer = self.hypothetical_chain.invoke(
            {"utterance": utterance}
        )
        return self.retriever.invoke(hypothetical_answer)

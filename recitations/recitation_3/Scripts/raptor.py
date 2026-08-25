

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic import NomicEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans

from Scripts.BaseRag import BaseRag
import os
os.environ["USER_AGENT"] = "MyPersonalBot/1.0"


class RAPTOR(BaseRag):
    def __init__(self, cluster_size=8, max_clusters=8, max_levels=3):
        self.cluster_size = cluster_size
        self.max_clusters = max_clusters
        self.max_levels = max_levels
        super().__init__()

    def indexing(self):
        docs = [WebBaseLoader(url, header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}).load()for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        leaf_docs = text_splitter.split_documents(docs_list)

        embedding = NomicEmbeddings(
            model="nomic-embed-text-v1.5", inference_mode="local"
        )
        all_nodes = list(leaf_docs)
        current_nodes = leaf_docs
        summarizer = ChatOllama(model="llama3.1", temperature=0)

        for level in range(self.max_levels):
            node_count = len(current_nodes)
            if node_count <= self.cluster_size:
                break

            vectors = embedding.embed_documents(
                [node.page_content for node in current_nodes]
            )

            # calulate the number of clusters. At least 2 clusters, at most max_cluters and as many cluster as nodes.
            target_clusters = (node_count + self.cluster_size - 1) // self.cluster_size
            cluster_count = min(max(2, target_clusters), self.max_clusters, node_count)

            labels = KMeans(n_clusters=cluster_count, random_state=0, n_init=10).fit_predict(vectors)

            next_nodes = []

            # loop through all labels and create nodes with summeries of the clusters
            for cluster_id in range(cluster_count):
                members = [
                    node for node, label in zip(current_nodes, labels)
                    if label == cluster_id
                ]
                if not members:
                    continue
                source_text = "\n\n".join(node.page_content for node in members)
                summary = summarizer.invoke(
                    "Summarize the following related passages. Preserve important "
                    "facts, names, dates, and relationships.\n\n" + source_text
                )
                next_nodes.append(
                    type(members[0])(
                        page_content=str(summary.content),
                        metadata={"raptor_level": level + 1, "cluster": cluster_id},
                    )
                )

            all_nodes.extend(next_nodes)
            current_nodes = next_nodes

        vectorstore = SKLearnVectorStore.from_documents(
            documents=all_nodes, embedding=embedding
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4})

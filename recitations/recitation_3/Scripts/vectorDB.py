from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings

class VectorDBClassifier:
    def __init__(self, training_set, test_set, good_reviews, k=6):
        if k < 1:
            raise ValueError("k must be at least 1")

        self.training_set = training_set
        self.test_set = test_set
        self.good_reviews = good_reviews

        embedding_model=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        #v1 = embedding_model.embed_query(test_set[0])

        vectorstore = SKLearnVectorStore.from_texts(
            texts=training_set,
            embedding=embedding_model
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    def classify(self):
        correct = 0
        for i, test in enumerate(self.test_set):
            if test in self.good_reviews:
                sales = True
            else:
                sales = False
            documents = self.retriever.invoke(test)
            score = 0
            n = 0
            for doc in documents:
                n +=1
                if sales and doc.page_content in self.good_reviews:
                    score += 1
                if not sales and doc.page_content not in self.good_reviews:
                    score += 1
            print(f"{test}:{score/n}")
            if score/n > 0.5:
                correct +=1
        return correct, len(self.test_set)



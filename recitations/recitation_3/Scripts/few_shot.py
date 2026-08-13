from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# from dealership import sales_questions, training_set, test_set
# from hotel_reviews import good_reviews, training_set, test_set


class FewShotClassifier:
    def __init__(self, training_set, test_set, good_reviews):
        self.training_set = training_set
        self.test_set = test_set
        self.good_reviews = good_reviews
        prompt = PromptTemplate(
            template="""The following are examples of accurate classification into one of the two categories: 
            good hotel review, bad hotel review \n\n {examples} \n\n
            What should {question} classify as? Answer directly, good or bad""",
            input_variables=["examples", "question"],
        )

        llm = OllamaLLM(
            model="llama3.1",
            temperature=0,
        )

        self.fewshot_chain = prompt | llm | StrOutputParser()

    def classify(self):

        examples = ""
        for i in range(0,10):
            examples += f"'{self.training_set[i]}' is a {'good' if self.training_set[i] in self.good_reviews else 'bad'} review.\n"

        correct = 0
        for i in range(0,len(self.test_set)):
            if self.test_set[i] in self.good_reviews:
                good = True
            else:
                good = False
            response = self.fewshot_chain.invoke(
                {"examples": examples, "question": self.test_set[i]}
            )

            if good and "good" in response.lower():
                correct += 1
            if not good and "good" not in response.lower():
                correct += 1

            print(f"{self.test_set[i]}:{response}")

        print(correct)
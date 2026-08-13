from email import message

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class OllamaChat:

    def __init__(self, prompt_text, bot_name = None, max_history_length=200):
        self.bot_name = bot_name
        prompt = PromptTemplate(template = prompt_text,
            input_variables=["history","utterance"],
        )

        llm = OllamaLLM(
            model="gemma3:4b",
            temperature=0,
        )

        self.turn_chain = prompt |llm | StrOutputParser()

        self.max_history_length = max_history_length
        self.history = ""

    def chat(self):

        while True:
            utterance = input(">")
            if utterance == "quit":
                print("Exiting chat...")
                break

            print(f"User: {utterance}")

            response = self.turn_chain.invoke(
                {"history": self.history,
                "utterance": utterance}
            )

            # If a name has been specified.
            if self.bot_name:
                print(f"{self.bot_name}: ", end="")
            print(f"{response}")
            self.history += "\n" + utterance + "\n" + response + "\n"
            self.history = self.history[-self.max_history_length:]

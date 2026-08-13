from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(template =
"""
You are an assistant for chatting with the user. 
Here is the history of the dialogue so far {history}.
Now the user says: {utterance}.  
    """,
    input_variables=["history","utterance"],
)

llm = OllamaLLM(
    model="gemma3:4b",
    temperature=0,
)

turn_chain = prompt | llm | StrOutputParser()

max_history_length = 200
history = ""

while True:
    utterance = input(">")
    if utterance == "quit":
        break

    response = turn_chain.invoke(
        {"history":history,
         "utterance": utterance}
    )
    print(f"{response}")
    history += "\n" + utterance + "\n" + response + "\n"
    history = history[-max_history_length:]

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from xml.dom.minidom import parse

import os
import time
import sqlite3

domain = "aircraft"

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spider", domain)

connection = sqlite3.connect(os.path.join(base_dir, f"{domain}.db"))

prompt = PromptTemplate(
    template="""
    Given the relational database represented in the XML: 
    {schema}
    Give the SQL in SQLite that answers the question: {question}
    Answer with only a JSON object with key sql that contains exactly the SQL and 
    the key paraphrase contains an English description of the question.
    """,
    input_variables=["question","schema"],
)

llm = OllamaLLM(
    model="llama3.1",
    temperature=0,
)

chain = prompt | llm | JsonOutputParser()

with open(os.path.join(base_dir, f"{domain}.cphrase")) as f:
    schema = f.read()
    print(f"{schema}")

dom = parse(os.path.join(base_dir, "train.corpus"))

correct = 0
total = 0
failed = 0

for qtag in dom.getElementsByTagName("question"):
    question = qtag.getAttribute("text")
    stag = qtag.getElementsByTagName("sql")[0]
    sql = stag.getAttribute("query")

    correct_answer = ""
    cursor = connection.cursor()
    cursor.execute(sql)
    for row in cursor:
        correct_answer += str(row)

    total += 1
    print(f"\nquery #{total}: question:{question}")
    print(f"correct SQL: {sql}")
    print(f"correct answer:{correct_answer}")

    start = time.time()
    try:
        response = chain.invoke(
            {"schema": schema,
             "question": question}
        )
    except Exception as e:
        print(f"{str(e)}")
        continue

    if not "paraphrase" in response:
        response["paraphrase"] = "FAILED TO CALCULATE PARAPHRASE"
    if not "sql" in response:
        response["sql"] = "FAILED TO CALCULATE SQL"

    print(f"Duration: {(time.time() - start):.2f} seconds")
    print(f"Paraphrase:{response['paraphrase']}")
    print(f"Calculated SQL:{response['sql']}")

    calculated_answer = ""

    try:
        cursor = connection.cursor()
        cursor.execute(response['sql'])
        for row in cursor:
            calculated_answer += str(row)

        if correct_answer == calculated_answer:
            print("correct!")
            correct += 1
        else:
            print("incorrect")
    except Exception as e:
        failed += 1
        print("failed: " + str(e))


print(f"total:{total} correct:{correct} failures:{failed} accuracy:{(correct / total):.2f}")



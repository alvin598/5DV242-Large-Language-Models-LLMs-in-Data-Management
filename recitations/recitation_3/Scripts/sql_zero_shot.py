from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from xml.dom.minidom import parse, parseString, Document

import os
import time
import sqlite3
import inflect

class TextToSQL:

    def __init__(self):
        

        self.base_dir = os.path.join("data", "spider")
        self.domains = os.listdir(self.base_dir)
        print(f"Found domains: {self.domains}")


        prompt = PromptTemplate(
            template="""
            Given the relational database represented in the XML: 
            {schema}
            Here are some example questions with their correct SQL:
            {examples}
            Give the SQL in SQLite that answers the question: {question}
            Answer with only a JSON object with key sql that contains exactly the SQL and 
            the key paraphrase contains an English description of the question.
            """,
            input_variables=["question","schema","examples"],
        )

        llm = OllamaLLM(
            model="llama3.1",
            temperature=0,
        )

        self.chain = prompt | llm | JsonOutputParser()

        self.inflector = inflect.engine()

    # normilize the words
    def normalize_word(self, word):
        word = word.lower().strip(".,!?;:")

        if not word:
            return word

        singular = self.inflector.singular_noun(word)  # type: ignore[arg-type]

        if singular:
            return singular

        return word


    def linked_schema(self, schema, question):
        schema_document = parseString(schema)
        tables = schema_document.getElementsByTagName("table")
        question_words = {
            self.normalize_word(word)
            for word in question.split()
        }

        linked_tables = []
        for table in tables:
            table_id = table.getAttribute("id")
            id_words = {
                self.normalize_word(word)
                for word in table_id.split("_")
            }

            if question_words.intersection(id_words):
                linked_tables.append(table)

        # no matching tables, return the original schema
        if not linked_tables:
            return schema

        result = Document()
        root = result.createElement("config")
        result.appendChild(root)

        for table in linked_tables:
            root.appendChild(table.cloneNode(deep=True))

        return result.toxml()

    def run(self, databases=3):

        overall_correct = 0
        overall_total = 0
        overall_failed = 0
        domain_results = {}

        count = 0
        for domain in self.domains:
            if count >= databases:
                break
            count += 1
            domain_dir = os.path.join(self.base_dir, domain)

            connection = sqlite3.connect(os.path.join(domain_dir, f"{domain}.db"))

            with open(os.path.join(domain_dir, f"{domain}.cphrase")) as f:
                schema = f.read()
                # print(f"{schema}")

            dom = parse(os.path.join(domain_dir, "train.corpus"))

            qtags = dom.getElementsByTagName("question")

            # for few-shot, include the first 3 qurstions as examples
            num_examples = 3
            examples = ""
            for qtag in qtags[:num_examples]:
                example_sql = qtag.getElementsByTagName("sql")[0].getAttribute("query")
                examples += f"Question: {qtag.getAttribute('text')}\nSQL: {example_sql}\n\n"

            correct = 0
            total = 0
            failed = 0

            for qtag in qtags[num_examples:]:
                question = qtag.getAttribute("text")
                stag = qtag.getElementsByTagName("sql")[0]
                sql = stag.getAttribute("query")

                correct_answer = ""
                cursor = connection.cursor()
                cursor.execute(sql)
                for row in cursor:
                    correct_answer += str(row)

                total += 1
                print(f"\n[{domain}] query #{total}: question:{question}")
                print(f"correct SQL: {sql}")
                print(f"correct answer:{correct_answer}")

                reduced_schema = self.linked_schema(schema, question)

                start = time.time()
                try:
                    response = self.chain.invoke(
                        {"schema": reduced_schema,
                        "question": question,
                        "examples": examples}
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

            connection.close()

            print(f"\n[{domain}] total:{total} correct:{correct} failures:{failed} "
                f"accuracy:{(correct / total if total else 0):.2f}")

            domain_results[domain] = (total, correct, failed)
            overall_total += total
            overall_correct += correct
            overall_failed += failed

        return domain_results, overall_total, overall_correct, overall_failed

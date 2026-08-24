from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from xml.dom.minidom import parse, parseString, Document

import os
import time
import sqlite3
import inflect
import random

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

        scope_prompt = PromptTemplate(
            template="""
            You are checking whether a database can answer a question or if it is out of scope.

            Database schema:
            {schema}

            Question:
            {question}

            Return only JSON:
            {{
                "in_scope": true or false,
                "reason": "short explanation"
            }}

            "in_scope" should be true if the question is in scope, otherwise false.
            """,
            input_variables=["schema", "question"],
        )

        summary_prompt = PromptTemplate(
            template="""
            A user asked: {question}
            Running the corresponding SQL query returned {row_count} rows:
            {rows}
            Write a short, plain-English summary of these results that answers the question.
            Do not invent rows or values that are not present in the results.
            """,
            input_variables=["question", "row_count", "rows"],
        )

        llm = OllamaLLM(
            model="llama3.1",
            temperature=0,
            format="json",
        )
        self.scope_chain = scope_prompt | llm | JsonOutputParser()
        self.chain = prompt | llm | JsonOutputParser()

        # instance of llm not with json
        summary_llm = OllamaLLM(model="llama3.1", temperature=0)
        self.summary_chain = summary_prompt | summary_llm

        # set the threshold for summarizing results
        self.summary_row_threshold = 10

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

        table_by_id = {}
        linked_ids = set()
        for table in tables:
            table_id = table.getAttribute("id")
            table_by_id[table_id] = table
            id_words = {
                self.normalize_word(word)
                for word in table_id.split("_")
            }

            if question_words.intersection(id_words):
                linked_ids.add(table_id)

        # add all linked tables by foreign keys
        for table_id in list(linked_ids):
            for fk in table_by_id[table_id].getElementsByTagName("foreign_key"):
                target = fk.getAttribute("to")
                if target in table_by_id:
                    linked_ids.add(target)

        # no matching tables, return the original schema
        if not linked_ids:
            print("No linked tables found, returning original schema.")
            return schema
        result = Document()
        root = result.createElement("config")
        result.appendChild(root)

        for table_id in linked_ids:
            root.appendChild(table_by_id[table_id].cloneNode(deep=True))

        return result.toxml()

    def summarize_results(self, question, rows):
        if not rows:
            return "The query returned no rows."

        return self.summary_chain.invoke({
            "question": question,
            "row_count": len(rows),
            "rows": "\n".join(str(row) for row in rows),
        })

    def in_scope(self, schema, question):
        scope_result = self.scope_chain.invoke({
            "schema": schema, 
            "question": question,
        })
        return scope_result

    def run(self, databases=3):

        overall_correct = 0
        overall_total = 0
        overall_failed = 0
        domain_results = {}

        # get a list of n random domains
        selected_domains = random.sample(self.domains, databases) if databases < len(self.domains) else self.domains

        count = 0
        for domain in selected_domains:
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


            # add a dummy question to test out-of-scope handling
            # --------------------------------------------------
            stupid_question = dom.createElement("question")
            stupid_question.setAttribute("text", "What is the meaning of life?")
            stupid_sql = dom.createElement("sql")
            stupid_sql.setAttribute("query", "SELECT 42;")
            stupid_question.appendChild(stupid_sql)
            qtags.append(stupid_question)
            # --------------------------------------------------

            # include the first 3 qurstions as examples for the few-shot
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
                    scope_result = self.in_scope(schema, question)
                    if not scope_result["in_scope"]:
                        failed += 1
                        print(f"out-of-scope: {scope_result.get('reason', 'no reason provided')}")
                        continue

                    response = self.chain.invoke(
                        {"schema": reduced_schema,
                        "question": question,
                        "examples": examples}
                    )
                except Exception as e:
                    failed += 1
                    print(f"generation failed ({type(e).__name__}): {e}")
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
                    rows = cursor.fetchall()
                    for row in rows:
                        calculated_answer += str(row)

                    if len(rows) > self.summary_row_threshold:
                        summary = self.summarize_results(question, rows)
                        print(f"Result summary ({len(rows)} rows): {summary}")

                    if correct_answer == calculated_answer:
                        print("correct!")
                        correct += 1
                    else:
                        failed += 1
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

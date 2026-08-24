from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from xml.dom.minidom import parse, parseString, Document

import os
import sqlite3
import inflect

class TextToSQL:

    def __init__(self):
        recitations_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.base_dir = os.path.join(recitations_dir, "recitation_3", "data", "spider")
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
        
        scope_prompt = PromptTemplate(
            template="""
            You are checking whether a database can answer a question.

            Database schema:
            {schema}

            Question:
            {question}

            Return only JSON:
            {{
                "in_scope": true or false,
                "reason": "short explanation"
            }}

            Return false if the requested information is not represented
            by any table or column in the schema. Do not use outside knowledge.
            """,
            input_variables=["schema", "question"],
        )
        self.scope_chain = scope_prompt | llm | JsonOutputParser()
        

        self.chain = prompt | llm | JsonOutputParser()

        self.inflector = inflect.engine()

    # normilize the words
    def normalize_word(self, word):
        word = word.lower().strip(".,!?;:")

        if not word:
            return word

        singular = self.inflector.singular_noun(word)

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

    def ask(self, database, question):
        domain_dir = os.path.join(self.base_dir, database)
        database_path = os.path.join(domain_dir, f"{database}.db")
        schema_path = os.path.join(domain_dir, f"{database}.cphrase")
        corpus_path = os.path.join(domain_dir, "train.corpus")

        with open(schema_path, encoding="utf-8") as schema_file:
            schema = schema_file.read()

        dom = parse(corpus_path)
        qtags = dom.getElementsByTagName("question")
        examples = ""
        for qtag in qtags[:3]:
            example_sql = qtag.getElementsByTagName("sql")[0].getAttribute("query")
            examples += (
                f"Question: {qtag.getAttribute('text')}\n"
                f"SQL: {example_sql}\n\n"
            )
        self.in_scope(schema, question)


        response = self.chain.invoke({
            "schema": self.linked_schema(schema, question),
            "question": question,
            "examples": examples,
        })
        generated_sql = response.get("sql")
        if not generated_sql:
            raise ValueError("The model did not return SQL.")

        with sqlite3.connect(database_path) as connection:
            cursor = connection.execute(generated_sql)
            columns = [description[0] for description in cursor.description or []]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return {
            "database": database,
            "question": question,
            "sql": generated_sql,
            "paraphrase": response.get("paraphrase", ""),
            "columns": columns,
            "rows": rows,
        }


    def in_scope(self, schema, question):
        scope_result = self.scope_chain.invoke({
            "schema": schema, 
            "question": question,
        })

        if not scope_result["in_scope"]:
            raise UserWarning("Question is outside the scope of the database.")

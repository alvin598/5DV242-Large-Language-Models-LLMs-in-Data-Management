import json
import os
import sys

from fastmcp import FastMCP

from sql_zero_shot import TextToSQL


text_to_sql = TextToSQL()
SUPPORTED_DATABASES = text_to_sql.domains
mcp = FastMCP(
	"Recitation 3 Text-to-SQL",
	instructions=(
		"This server answers natural-language questions over every local SQLite "
		"database in the Spider knowledge base. Supported databases: "
		+ ", ".join(SUPPORTED_DATABASES)
	),
)

@mcp.tool()
def list_supported_databases() -> list[str]:
	"""List every SQLite database available in the Spider knowledge base."""
	return SUPPORTED_DATABASES

@mcp.tool()
def ask_sql_question(database: str, question: str) -> str:
	"""Generate and execute SQLite SQL for a question on one supported database."""
	try:
		result = text_to_sql.ask(database, question)
		return json.dumps(result, ensure_ascii=True)
	except Exception as e:
		raise ValueError(f"Error processing question: {e}")

if __name__ == "__main__":
	mcp.run(transport="stdio")

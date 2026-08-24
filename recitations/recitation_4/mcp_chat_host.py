import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def result_text(result: object) -> str:
	return "\n\n".join(
		getattr(content, "text", str(content)) for content in result.content
	)


async def chat():
	server_path = Path(__file__).with_name("mcp_server.py")
	server = StdioServerParameters(
		command=sys.executable,
		args=[str(server_path)],
		cwd=str(server_path.parent),
	)

	async with stdio_client(server) as (read_stream, write_stream):
		async with ClientSession(read_stream, write_stream) as session:
			await session.initialize()
			databases_result = await session.call_tool("list_supported_databases")
			databases = json.loads(result_text(databases_result))

			print("Connected to the Spider database server. Type 'quit' to exit.")
			while True:
				database = input("Database name:  \n").strip()
				if database.lower() == "quit":
					return
				if database.lower() == "list":
					print(", ".join(databases))
					continue
				if database not in databases:
					print("Out of scope: choose one of the supported Spider databases.")
					continue

				question = input("Question: ").strip()
				if not question:
					continue

				try:
					answer = await session.call_tool(
						"ask_sql_question",
						arguments={"database": database, "question": question},
					)
				except Exception as e:
					print(f"Error: {e}")
					continue

				print(f"Answer: {result_text(answer)}")


if __name__ == "__main__":
	asyncio.run(chat())

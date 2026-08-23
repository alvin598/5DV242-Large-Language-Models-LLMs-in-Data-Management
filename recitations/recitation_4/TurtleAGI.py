import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def turtleAGI(objective):

    t_id = 1
    initial_task = {
        "task_id": t_id,
        "task_depth": 0,
        "task_name": "Develop a task list"
    }

    agenda = [initial_task]

    while True:
        if not agenda:
            break

        task = agenda.pop(0)

        print(f"NEXT TASK: {task['task_name']}, DEPTH: {task['task_depth']}\n")

        raw_solution = generate_raw_solution(objective, task["task_name"])
        task_names = list(map(lambda x:x['task_name'],agenda))

        if task["task_depth"] < 2:
            new_tasks = normalize_solution(
                objective,
                raw_solution,
                task["task_name"],
                task_names
            )
        else:
            new_tasks = []

        for new_task in new_tasks:
            t_id +=1
            new_task['task_id'] = t_id
            new_task['task_depth'] = task["task_depth"] + 1
            agenda.append(new_task)

        prioritized_tasks = prioritize_solution(objective, agenda)

        if prioritized_tasks:
            agenda = prioritized_tasks

def generate_raw_solution(objective, task):
    prompt = f'Perform the following task based on the following objective: {objective}.\n'
    prompt += f'\n task: {task}\nResponse:'
    return llm_call(prompt, max_len=2000)

def normalize_solution(objective, result, task_description, task_list):
    prompt = f"""
You are to use the raw task results to create new tasks with the following objective: {objective}.
The last completed task has the raw result: \n{result}
This result was based on this task description: {task_description}.\n"""

    if task_list:
        prompt += f"These are incomplete tasks: {', '.join(task_list)}\n"
    prompt += "Based on the result, return a list of tasks to be completed in order to meet the objective. "
    if task_list:
        prompt += "These new tasks must not overlap with incomplete tasks. "

    prompt += """
Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period. If your list is empty, write "There are no tasks to add at this time."
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output."""

    response = llm_call(prompt, max_len=2000)
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)

    rv = [{"task_name": task_name} for task_name in new_tasks_list]
    return rv

def prioritize_solution(objective, agenda):

    task_names = [task["task_name"] for task in agenda]

    task_material = '\n' + '\n'.join(task_names)

    prompt = f"""
You are tasked with prioritizing the following tasks: {task_material}
Consider the ultimate objective of your team: {objective}.
Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites or are more essential for meeting the objective.
Do not remove any tasks. Return the ranked tasks as a numbered list in the format:

#. First task
#. Second task

The entries must be consecutively numbered, starting with 1. The number of each entry must be followed by a period.
Do not include any headers before your ranked list or follow your list with any other output."""


    response = llm_call(prompt, max_len=2000)
    if not response:
        print('Received empty response from priotritization agent. Keeping task list unchanged.')
        return
    ranked_task_names = []
    for task_string in response.split("\n"):
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip():
                ranked_task_names.append(task_name)

    tasks_by_name = {task["task_name"]: task for task in agenda}
    return [
        tasks_by_name[task_name]
        for task_name in ranked_task_names
        if task_name in tasks_by_name
    ]


llm = ChatOllama(
    model="gemma3",
    temperature=0,
)

prompt_object = PromptTemplate(
        template="{prompt}",
        input_variables=["prompt"],
    )

chain = prompt_object | llm | StrOutputParser()

def llm_call(prompt, max_len=2000):
    response = chain.invoke({"prompt": prompt})
    return response[:max_len]

if __name__ == "__main__":
    turtleAGI(input("what is the objective? (e.g. solve world hunger)"))
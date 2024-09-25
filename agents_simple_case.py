import ollama
from langchain.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory

# Initialize Ollama model
llm = Ollama(model="gemma2:2b")

# Initialize task list
tasks = []

# Define tools
def add_task(task):
    tasks.append({"task": task, "completed": False})
    return f"Task added: {task}"

def list_tasks(*param):
    print(param)
    if not tasks:
        return "No tasks found."
    return "\n".join([f"{i+1}. {'[x]' if t['completed'] else '[ ]'} {t['task']}" for i, t in enumerate(tasks)])

def complete_task(task_number):
    try:
        index = int(task_number) - 1
        if 0 <= index < len(tasks):
            tasks[index]["completed"] = True
            return f"Task {task_number} marked as complete."
        else:
            return f"Invalid task number: {task_number}"
    except ValueError:
        return f"Invalid input: {task_number}. Please provide a valid task number."

tools = [
    Tool(name="Add Task", func=add_task, description="Add a new task to the list"),
    Tool(name="List Tasks", func=list_tasks, description="List all tasks"),
    Tool(name="Complete Task", func=complete_task, description="Mark a task as complete by providing its number"),
]

# Initialize agent
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)

# Main interaction loop
print("Task Management Agent (type 'quit' to exit)")
while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        break
    response = agent.run(user_input)
    print("Agent:", response)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from datetime import datetime
import os
import ast
import operator

# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")

# ==========================================
# OpenAI LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ==========================================
# Calculator Logic
# ==========================================

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def safe_calculate(expression):

    def evaluate(node):

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            operator_type = type(node.op)
            if operator_type not in allowed_operators:
                raise ValueError("Operator not allowed")
            return allowed_operators[operator_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            operator_type = type(node.op)
            if operator_type not in allowed_operators:
                raise ValueError("Operator not allowed")
            return allowed_operators[operator_type](operand)

        raise ValueError("Invalid expression")

    parsed = ast.parse(expression, mode="eval")
    return evaluate(parsed.body)

# ==========================================
# Tools
# ==========================================

@tool
def calculator(expression: str) -> str:
    """Use this tool for math calculations."""
    try:
        result = safe_calculate(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def current_time(dummy_input: str = "") -> str:
    """Use this tool when the user asks for the current time or date."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %I:%M:%S %p")


@tool
def read_file(filename: str) -> str:
    """
    Use this tool whenever the user wants to read, open, view, summarize,
    analyze, or explain the contents of a local text file.
    Input should be a filename like: notes.txt, report.txt, data.txt
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading file: {e}"

# ==========================================
# Create Agent (modern approach)
# ==========================================

tools = [calculator, current_time, read_file]

agent = create_react_agent(llm, tools)

# ==========================================
# Chat Loop
# ==========================================

print("LangChain Agent Started!")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    try:
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
        response = result["messages"][-1].content
        print("\nBot:", response)
        print()

    except Exception as e:
        print("\nError:", e)
        print()
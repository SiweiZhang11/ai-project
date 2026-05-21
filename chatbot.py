from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import ast
import operator
import json

# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

# ==========================================
# OpenAI Client
# ==========================================

client = OpenAI(api_key=api_key)

MODEL_NAME = "gpt-4o-mini"

# ==========================================
# Calculator Tool
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


def calculator_tool(expression):

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

    parsed_expression = ast.parse(expression, mode="eval")

    return evaluate(parsed_expression.body)

# ==========================================
# Current Time Tool
# ==========================================


def current_time_tool():

    now = datetime.now()

    return now.strftime("%Y-%m-%d %I:%M:%S %p")

# ==========================================
# ONE AI REQUEST FOR TOOL DECISION
# ==========================================


def decide_action(user_input):

    prompt = f"""
You are an AI agent router.

Your job is to decide:
1. Which tool should be used
2. What arguments are needed

Available tools:

1. calculator
Use for:
- math
- calculations
- percentages

Example:
"What is 25 times 18?"

Return:
{{
    "tool": "calculator",
    "expression": "25 * 18"
}}

ONLY use this tool when the user wants the ACTUAL live current time/date.

Use for:
- "What time is it?"
- "Current date please"
- "Tell me today's date"

DO NOT use for:
- explanations
- educational questions
- conceptual discussions
- questions ABOUT time calculation

Examples:

User:
"What time is it?"
Return:
{{
    "tool": "current_time"
}}

User:
"What is today's date?"
Return:
{{
    "tool": "current_time"
}}

User:
"How do clocks work?"
Return:
{{
    "tool": "none"
}}

User:
"How do you calculate the current time?"
Return:
{{
    "tool": "none"
}}

User:
"Explain time zones"
Return:
{{
    "tool": "none"
}}

IMPORTANT:
Return ONLY valid JSON.
No markdown.
No explanation.

User message:
{user_input}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = response.choices[0].message.content.strip()

    return json.loads(text)

# ==========================================
# Normal Chat
# ==========================================


def normal_chat(user_input, chat_history):

    chat_history.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=chat_history
    )

    bot_reply = response.choices[0].message.content

    chat_history.append({
        "role": "assistant",
        "content": bot_reply
    })

    return bot_reply

# ==========================================
# Chatbot Loop
# ==========================================

print("OpenAI Agent Chatbot Started!")
print("Type 'exit' to quit.\n")

chat_history = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:

        # ==========================================
        # ONE AI DECISION
        # ==========================================

        decision = decide_action(user_input)

        print("\n[AI Decision]")
        print(decision)

        tool = decision["tool"]

        # ==========================================
        # Calculator Tool
        # ==========================================

        if tool == "calculator":

            expression = decision["expression"]

            result = calculator_tool(expression)

            print(f"\nBot: The result is {result}\n")

        # ==========================================
        # Current Time Tool
        # ==========================================

        elif tool == "current_time":

            current_time = current_time_tool()

            print(f"\nBot: Current date/time is {current_time}\n")

        # ==========================================
        # Normal Chat
        # ==========================================

        else:

            bot_reply = normal_chat(user_input, chat_history)

            print(f"\nBot: {bot_reply}\n")

    except Exception as e:

        print("\nError:", e, "\n")
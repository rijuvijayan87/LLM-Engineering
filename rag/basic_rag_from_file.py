import glob
import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from gradio import gradio as gr

knowledge = {}


def fetch_employee_details():
    employee_file_list = glob.glob("rag/knowledge-base/employees/*")

    for employee_file in employee_file_list:
        file_name = Path(employee_file).stem
        last_name = file_name.split(" ")[-1]

        with open(employee_file, "r") as f:
            knowledge[last_name] = f.read()


def fetch_product_details():
    product_file_list = glob.glob("rag/knowledge-base/products/*")

    for product_file in product_file_list:
        file_name = Path(product_file).stem

        with open(product_file, "r") as f:
            knowledge[file_name] = f.read()


def get_openai_key():
    load_dotenv(override=True)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        print("Error: OPENAI_API_KEY environment variable not set.")
        exit(1)

    return openai_key


def call_llm(model, messages):
    if model is None:
        raise ValueError("model is not provided")

    if messages is None:
        raise ValueError("messages is not provided")

    openai.api_key = get_openai_key()

    stream = openai.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


def get_user_prompt(user_input):
    if user_input is None:
        raise ValueError("user_input is not provided")

    return [{"role": "user", "content": user_input}]


def get_system_template():
    return f"""
        You are an amazing chatbot assistant at InsuranceLLM. You are provided with context about the 
        employee information of the firm and the products the firm has developed. 
        Information are quite detailed. Only use the context provided to answer the question. if you are asked 
        to do calculations on total number of employees, total salaries paid every year, total bonus paid every year etc
        you should help calculating these details.
        If you do not know answer to the question asked, please say so.

        Additional context as follows:
    """


def chat_gr(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]

    messages = (
        [
            {
                "role": "system",
                "content": f"{get_system_template()}\n additional context about the employees and products: {knowledge}",
            }
        ]
        + history
        + [{"role": "user", "content": message}]
    )

    model = "gpt-4o-mini"
    for response_chunk in call_llm(model=model, messages=messages):
        yield response_chunk


def main():
    """Main function to get API key and call OpenAI."""
    fetch_employee_details()
    fetch_product_details()

    print(f"total knowledge object {len(knowledge)}")
    print(f"keys -> {knowledge.keys()}")

    gr.ChatInterface(fn=chat_gr).launch(inbrowser=True)


if __name__ == "__main__":
    main()

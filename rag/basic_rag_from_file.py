import glob
import os
from pathlib import Path
from typing import List

import gradio as gr
import openai
from dotenv import load_dotenv


class ChatBot:
    """
    A chatbot that uses the context that is available in the form of documents
    to answer questions
    """

    def __init__(self, knowledge_base_paths: list[str], model="gpt-4o-mini"):
        """
        Initializes the ChatBot.

        Args:
            knowledge_base_paths: A list of glob patterns for knowledge base files.
            model: The OpenAI model to use for chat completions.

        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set.
        """
        self.model = model
        load_dotenv(override=True)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Error: OPENAI_API_KEY environment variable not set!")
        openai.api_key = api_key

        self._knowledge_base = self._build_knowledge_base(knowledge_base_paths)

    def _build_knowledge_base(self, knowledge_base_paths: list[str]) -> dict[str, str]:
        """
        Builds the knowledge base from files matching the provided glob patterns.
        The key for the knowledge base is derived from the filename.
        """
        knowledge = {}
        for knowledge_base_path in knowledge_base_paths:
            file_list = glob.glob(knowledge_base_path)

            for file in file_list:
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():  # Ensure content is not just whitespace
                            stem = Path(file).stem
                            key = stem.split(" ")[-1]
                            knowledge[key] = content
                except IOError as e:
                    print(f"Warning: Could not read file {file}: {e}")
        return knowledge

    def get_system_template(self) -> str:
        return f"""
            You are an amazing chatbot assistant at InsuranceLLM. You are provided with context about the 
            employee information of the firm and the products the firm has developed. 
            Information are quite detailed. Only use the context provided to answer the question. if you are asked 
            to do calculations on total number of employees, total salaries paid every year, total bonus paid every year etc
            you should help calculating these details.
            If you do not know answer to the question asked, please say so.

            Additional context as follows:
            {self._knowledge_base}
        """

    def _call_llm(self, messages, model):
        if model is None:
            raise ValueError("model value is not provided")

        if messages is None:
            raise ValueError("messages value is not provided")

        try:
            stream = openai.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            response = ""
            for chunk in stream:
                response += chunk.choices[0].delta.content or ""
                yield response
        except openai.APIConnectionError as e:
            error_message = f"Error: Failed to connect to OpenAI API. {e}"
            print(error_message)
            yield error_message
        except openai.RateLimitError as e:
            error_message = f"Error: Rate limit exceeded. {e}"
            print(error_message)
            yield error_message
        except (openai.AuthenticationError, openai.APIError) as e:
            error_message = f"Error: OpenAI API error. {e}"
            print(error_message)
            yield error_message

    def chat(self, message, history):
        """
        Main chat function to be used with Gradio.
        It takes a user message and conversation history, formats the prompt,
        and streams the response from the LLM.
        """
        formatted_history = [
            {"role": h["role"], "content": h["content"]} for h in history
        ]

        messages = (
            [{"role": "system", "content": self.get_system_template()}]
            + formatted_history
            + [{"role": "user", "content": message}]
        )

        yield from self._call_llm(messages=messages, model=self.model)

    def launch_gradio_interface(self):
        """
        Launches the Gradio chat interface for the bot.
        """
        gr.ChatInterface(
            fn=self.chat,
            chatbot=gr.Chatbot(height=400),
            title="InsuranceLLM Chat",
            description="Ask me questions about employees and products.",
        ).launch()


def main():
    """
    Driver Function
    """
    knowledge_base_paths = [
        "rag/knowledge-base/employees/*",
        "rag/knowledge-base/products/*",
    ]

    try:
        bot = ChatBot(knowledge_base_paths=knowledge_base_paths, model="gpt-4o-mini")
        bot.launch_gradio_interface()
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()

import openai
import requests

# Set up OpenAI API key
OPENAI_API_KEY = "your_openai_api_key"

# Step 1: Define the agent
class AIAgent:
    def __init__(self):
        self.tools = {
            "search": self.web_search,
            "llm": self.ask_llm
        }

    def decide_tool(self, query):
        """Decide whether to use web search or LLM."""
        if "latest" in query.lower() or "current" in query.lower():
            return "search"
        return "llm"

    def web_search(self, query):
        """Perform a web search (using a placeholder function)."""
        print(f"Searching the web for: {query}")
        return f"Fake search results for: {query}"  # Replace with real web search API

    def ask_llm(self, query):
        """Ask the LLM for an answer."""
        print(f"Querying LLM for: {query}")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": query}],
            api_key=OPENAI_API_KEY
        )
        return response["choices"][0]["message"]["content"]

    def execute(self, query):
        """Run the agent to find the best response."""
        tool = self.decide_tool(query)
        result = self.tools[tool](query)

        # Process and summarize response using LLM
        summary_prompt = f"Summarize this information in a few sentences: {result}"
        summary = self.ask_llm(summary_prompt)
        return summary

# Step 2: Use the AI Agent
agent = AIAgent()
query = "What are the latest AI trends?"
response = agent.execute(query)

print("\nFinal Response:", response)

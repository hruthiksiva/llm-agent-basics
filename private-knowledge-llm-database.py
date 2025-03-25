import sqlite3
import openai

# OpenAI API Key (Replace with your actual API key)
OPENAI_API_KEY = "your_openai_api_key"

class PrivateLLMAgent:
    def __init__(self, db_path):
        self.db_path = db_path

    def query_database(self, user_query):
        """Search private database for relevant information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Simple example: Searching a 'knowledge' table
        cursor.execute("SELECT info FROM knowledge WHERE topic LIKE ?", ('%' + user_query + '%',))
        result = cursor.fetchall()

        conn.close()
        return " ".join(row[0] for row in result) if result else "No relevant data found."

    def ask_llm(self, user_query, private_data):
        """Query LLM with private knowledge as context."""
        prompt = f"""
        You are an AI assistant with private knowledge.
        Here is private data: {private_data}
        User question: {user_query}
        Answer using both private knowledge and general knowledge.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            api_key=OPENAI_API_KEY
        )
        return response["choices"][0]["message"]["content"]

    def execute(self, user_query):
        """Fetch private data, inject into LLM, and generate a response."""
        private_data = self.query_database(user_query)
        response = self.ask_llm(user_query, private_data)
        return response

# Example Usage
agent = PrivateLLMAgent("private_data.db")  # SQLite Database
query = "What is the company's 2024 strategy?"
response = agent.execute(query)

print("\nFinal Response:", response)

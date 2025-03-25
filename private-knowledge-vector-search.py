import sqlite3
import faiss
import openai
import numpy as np

# OpenAI API Key
OPENAI_API_KEY = "your_openai_api_key"


# store knowledge

class PrivateKnowledgeBase:
    def __init__(self, db_path, faiss_index_path="knowledge.index"):
        self.db_path = db_path
        self.index_path = faiss_index_path
        self.dimension = 1536  # OpenAI embedding size (text-embedding-ada-002)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.data_store = []  # Store text along with vectors

    def get_embedding(self, text):
        """Convert text into an embedding vector using OpenAI."""
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text,
            api_key=OPENAI_API_KEY
        )
        return np.array(response["data"][0]["embedding"], dtype=np.float32)

    def store_knowledge(self):
        """Fetch private data, generate embeddings, and store in FAISS."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, info FROM knowledge")  # Table with private data
        rows = cursor.fetchall()
        conn.close()

        for row_id, text in rows:
            vector = self.get_embedding(text)
            self.index.add(np.array([vector]))  # Add to FAISS
            self.data_store.append((row_id, text))

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        print("Knowledge stored successfully!")

# Run this once to store knowledge
knowledge_base = PrivateKnowledgeBase("private_data.db")
knowledge_base.store_knowledge()




#retreive knowledge

class PrivateLLMAgent:
    def __init__(self, db_path, faiss_index_path="knowledge.index"):
        self.db_path = db_path
        self.index = faiss.read_index(faiss_index_path)
        self.data_store = []  # Store text for retrieval

    def get_embedding(self, text):
        """Convert text into a query embedding."""
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text,
            api_key=OPENAI_API_KEY
        )
        return np.array(response["data"][0]["embedding"], dtype=np.float32)

    def retrieve_knowledge(self, query, top_k=3):
        """Find the most relevant private knowledge using FAISS vector search."""
        query_vector = self.get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        retrieved_texts = []
        for idx in indices[0]:
            if idx < len(self.data_store):  # Ensure index is valid
                retrieved_texts.append(self.data_store[idx][1])  # Fetch original text

        return " ".join(retrieved_texts)

    def ask_llm(self, user_query, private_data):
        """Use retrieved private knowledge in an LLM response."""
        prompt = f"""
        You have access to private data:
        {private_data}
        User's question: {user_query}
        Answer based on this private knowledge.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert assistant."},
                      {"role": "user", "content": prompt}],
            api_key=OPENAI_API_KEY
        )
        return response["choices"][0]["message"]["content"]

    def execute(self, user_query):
        """Retrieve private knowledge and generate an AI response."""
        private_data = self.retrieve_knowledge(user_query)
        response = self.ask_llm(user_query, private_data)
        return response

# Example Usage
agent = PrivateLLMAgent("private_data.db")
query = "What are the latest business strategies for 2024?"
response = agent.execute(query)

print("\nFinal Response:", response)

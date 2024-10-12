import os
import json
import numpy as np
import faiss
import ollama
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load data
output_dir = 'traditional_rag_output'
with open(os.path.join(output_dir, 'chunks.json'), 'r') as f:
    chunks = json.load(f)

index = faiss.read_index(os.path.join(output_dir, 'faiss_index.bin'))

# Initialize Ollama LLM for text generation
llm = Ollama(
    model="llama3.2:3b",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

def generate_embedding(text):
    # Use Ollama to generate embeddings with nomic-embed-text
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return np.array(response['embedding'])

def retrieve_relevant_chunks(query, top_k=3):
    query_vector = generate_embedding(query)
    scores, indices = index.search(np.array([query_vector]).astype('float32'), top_k)
    return [chunks[i] for i in indices[0]]

def answer_query(query):
    relevant_chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""Based on the following context, please answer the query. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Query: {query}

Answer:"""

    return llm(prompt)

def main():
    print("Welcome to the Two Pups Pizza Traditional RAG Q&A system!")
    print("Ask any question about Two Pups Pizza, or type 'quit' to exit.")
    
    while True:
        user_query = input("Enter your query (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        answer = answer_query(user_query)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
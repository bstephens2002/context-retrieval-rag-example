import os
import re
import json
import faiss
import numpy as np
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import ollama

# Initialize Ollama LLM for text generation
llm = Ollama(
    model="llama3.2:3b",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

def generate_embedding(text):
    # Use Ollama to generate embeddings with nomic-embed-text
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return np.array(response['embedding'])

def create_index(enhanced_chunks):
    texts = [f"search_document: {chunk['context']}\n\n{chunk['original_chunk']}" for chunk in enhanced_chunks]
    embeddings = [generate_embedding(text) for text in texts]
    embeddings = np.array(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index, embeddings

def retrieve_relevant_chunks(query, index, enhanced_chunks, k=3):
    query_vector = generate_embedding(f"search_query: {query}")
    scores, indices = index.search(np.array([query_vector.astype('float32')]).reshape(1, -1), k)
    return [enhanced_chunks[i] for i in indices[0]]

def save_data(enhanced_chunks, index, embeddings, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save enhanced chunks
    with open(os.path.join(output_dir, 'enhanced_chunks.json'), 'w') as f:
        json.dump(enhanced_chunks, f)
    
    # Save index
    faiss.write_index(index, os.path.join(output_dir, 'faiss_index.bin'))
    
    # Save embeddings
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)

def answer_query(query, index, enhanced_chunks):
    relevant_chunks = retrieve_relevant_chunks(query, index, enhanced_chunks)
    context = "\n\n".join([f"{chunk['context']}\n\n{chunk['original_chunk']}" for chunk in relevant_chunks])
    
    prompt = f"""Based on the following context, please answer the query. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Query: {query}

Answer:"""

    return llm(prompt)

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_enhanced_chunks(chunks, document):
    enhanced_chunks = []
    for chunk in chunks:
        context = generate_context(chunk, document)
        enhanced_chunks.append({
            "context": context,
            "original_chunk": chunk
        })
    return enhanced_chunks

def generate_context(chunk, document):
    prompt = f"""
    Given the following chunk from a document about Two Pups Pizza, provide a concise context (50-100 tokens) that explains:
    1. The specific topic or category this chunk belongs to (e.g., menu item, pricing, dietary options, etc.)
    2. Any unique or important details about Two Pups Pizza mentioned in this chunk
    3. How this information might be relevant to customer inquiries or orders

    Chunk:
    {chunk}

    Context:
    """
    
    response = llm(prompt)
    return response.strip()

def main():
    input_file = 'data_Two_Pups_Pizza.md'
    output_dir = 'contextual_retrieval_output'
    
    document = load_document(input_file)
    chunks = split_into_chunks(document)
    enhanced_chunks = create_enhanced_chunks(chunks, document)
    
    index, embeddings = create_index(enhanced_chunks)
    save_data(enhanced_chunks, index, embeddings, output_dir)
    
    print(f"Processing complete. Output saved to {output_dir}")
    
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        answer = answer_query(query, index, enhanced_chunks)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
import os
import json
import faiss
import numpy as np
import ollama

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

def generate_embedding(text):
    # Use Ollama to generate embeddings with nomic-embed-text
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return np.array(response['embedding'])

def create_index(chunks):
    embeddings = [generate_embedding(chunk) for chunk in chunks]
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index, embeddings

def save_data(chunks, index, embeddings, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chunks
    with open(os.path.join(output_dir, 'chunks.json'), 'w') as f:
        json.dump(chunks, f)
    
    # Save index
    faiss.write_index(index, os.path.join(output_dir, 'faiss_index.bin'))
    
    # Save embeddings
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)

def main():
    input_file = 'data_Two_Pups_Pizza.md'  # Replace with your input file path
    output_dir = 'traditional_rag_output'
    
    # Load and process the document
    document = load_document(input_file)
    chunks = split_into_chunks(document)
    
    # Create and save the index and embeddings
    index, embeddings = create_index(chunks)
    save_data(chunks, index, embeddings, output_dir)
    
    print(f"Processing complete. Output saved to {output_dir}")

if __name__ == "__main__":
    main()
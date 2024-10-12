# context-retrieval-rag-example

# 📚 Two Pups Pizza Q&A System Example

Welcome to the Two Pups Pizza Q&A System example! This project demonstrates different approaches to building a question-answering system using Retrieval Augmented Generation (RAG) techniques.

## 🌟 Features

- **Contextual Retrieval:** Utilizes contextualized text chunks for improved query understanding.
- **Traditional RAG:** Implements a basic RAG approach using Ollama for embeddings and retrieval.
- **Ollama Integration:** Demonstrates the use of Ollama API for language model interactions.

## 🛠️ Implementation

The repository contains the following key components:

- `context-generator-v3.py`: Generates contextualized chunks from the input document.
- `context_retrieval_chat_v2.py`: Implements a chat interface for contextual retrieval.
- `traditional_rag_chat_v2.py`: Provides a chat interface for traditional RAG.
- `traditional_rag_generator_v2.py`: Creates chunks, index, and embeddings for traditional RAG.

## 🚀 Usage

Follow these steps to run the example:

1. **Install Dependencies:** 
   ```
   pip install -r requirements.txt
   ```

2. **Generate RAG Components:** 
   ```
   python traditional_rag_generator_v2.py
   ```
   This processes the input file (data_Two_Pups_Pizza.md) to create necessary components.

3. **Start the Chat Interface:** 
   Choose either:
   ```
   python context_retrieval_chat_v2.py
   ```
   or
   ```
   python traditional_rag_chat_v2.py
   ```
   to begin interacting with the Q&A system.

## 🎓 Educational Purpose

This project serves as an educational example to demonstrate:
- Implementation of different RAG techniques
- Integration of language models in Q&A systems
- Comparison between contextual and traditional retrieval methods

Feel free to explore, modify, and learn from this example! If you have any questions, check out the accompanying YouTube video for more detailed explanations.

Happy learning! 🧠💡
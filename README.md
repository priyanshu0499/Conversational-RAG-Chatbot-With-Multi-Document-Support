

# Conversational RAG Chatbot with Multi-Document Support

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built using Python, Streamlit, LangChain, and OpenAI’s **GPT-4o**. It enables users to interactively query or summarize content from various document types (PDF, DOCX, TXT, CSV). The chatbot uses embeddings to retrieve relevant document sections and provides accurate, context-sensitive responses based on user queries.

## Features

- **Multi-Document Support**: Supports PDF, DOCX, TXT, and CSV files for upload and processing.
- **Question-Answering (QA)**: Accurately answers questions based on relevant sections from uploaded documents.
- **Adaptive Summarization**: Offers customizable summaries based on user-specified length (short or detailed).
- **History-Aware Retrieval**: Uses context from prior queries within a session for more accurate question reformulation.
- **Embeddings and Vector Search**: Uses OpenAI embeddings to perform similarity-based document retrieval with Chroma.
- **Session Management**: Manages conversation history to provide continuity across interactions.

## Tech Stack

- **Python**: Core language.
- **Streamlit**: User interface for uploading documents and interacting with the chatbot.
- **LangChain**: Framework for language model-powered applications.
- **OpenAI GPT-4o**: Used for generating responses based on document context.
- **Chroma**: For vector storage and similarity-based document retrieval.
- **dotenv**: For environment variable management (loading API keys securely).
- **Document Processing Libraries**: `pdfplumber` for PDFs, `Docx2txtLoader` for DOCX, `pandas` for CSV.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Rename `.env.example` to `.env`.
   - Add your **OpenAI API Key** to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

1. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload Documents**:
   - Choose files in PDF, DOCX, TXT, or CSV formats.
   - Multiple files can be uploaded simultaneously, and they will be processed together for cross-document queries.

3. **Ask Questions or Request Summaries**:
   - Type your questions or summarization requests into the input box.
   - For summaries, specify "detailed" or "short" in your query to control summary length.

## System Architecture

The chatbot's architecture leverages a **Retrieval-Augmented Generation (RAG)** approach, illustrated below:

### RAG Workflow

1. **Document Ingestion**: Uploaded documents are loaded and chunked into manageable pieces.
2. **Embedding Generation**: Chunks are converted into embeddings using OpenAI embeddings.
3. **Contextual Question Reformulation**: If a question references past context, it is reformulated to make it standalone.
4. **Relevant Document Retrieval**: The reformulated or original question is used to retrieve relevant document chunks.
5. **Response Generation**: GPT-4o processes the retrieved context to answer the question or generate a summary.

```mermaid
graph LR
    A[Upload Document] --> B[Chunk Document]
    B --> C[Generate Embeddings]
    C --> D[Contextual Question Reformulation]
    D --> E[Retrieve Relevant Context]
    E --> F[Generate Response with GPT-4o]
    F --> G[Display Answer or Summary]
```

## Code Overview

### Key Components

- **Document Upload & Processing**: The code supports PDF, DOCX, TXT, and CSV formats. Each document type is processed with specific libraries for compatibility.
- **History-Aware Retrieval Chain**: Uses a **history-aware retriever** to incorporate context from previous user interactions for more accurate question reformulation.
- **Question-Answering and Summarization Prompts**: Customized prompts allow for both targeted question-answering and length-controlled summaries.
- **Embeddings and Vector Search**: Embeddings are generated with OpenAI's text-embedding-ada-002 model and stored in Chroma for efficient similarity search.
- **Response Generation**: GPT-4o uses retrieved content to provide answers or summaries based on user queries.

### Code Details

The main code structure includes:
- **QA Prompt**: Designed to answer questions based on relevant document sections.
- **Summarization Prompt**: Provides length-adjusted summaries based on user preference.
- **Contextual Question Prompt**: Reformulates user questions for context awareness.
- **Session State Management**: Maintains chat history and responses across sessions.

## Example Usage

- **Question**: "What are the main points discussed in the document?"
- **Summarization**: "Give a detailed summary of the document."
- **Question with Context**: "Can you explain more about the process?" (After a related question has been asked)

## Security

The `.env` file contains sensitive information like API keys. Make sure this file is **never uploaded to GitHub** by adding it to `.gitignore`.

## Contribution Guidelines

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License

This project is licensed under the MIT License.


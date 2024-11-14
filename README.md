Here's a structured, detailed `README.md` for your GitHub project:

---

# Conversational RAG with Document QA and Summarization

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit** that provides **question-answering (QA)** and **summarization** features across multiple document formats. Users can upload PDF, DOCX, TXT, or CSV files and interact with the content through conversational Q&A or generate tailored summaries.

## Features

- **Conversational Q&A**:
  - Answers questions based on the content of uploaded documents, retrieving relevant sections and generating concise, accurate answers.
  - Supports follow-up questions by providing history-aware retrieval, ensuring context across conversations.
  
- **Summarization**:
  - Provides summaries with adjustable length, including options for short summaries (focused on main points) and detailed summaries (in-depth overview).
  
- **Multi-Format Support**:
  - Uploads and processes PDF, DOCX, TXT, and CSV files.
  
## Tech Stack

- **Python**
- **Streamlit**: For UI and interactive document upload.
- **LangChain**: For chaining language model prompts and retrieval.
- **Chroma**: Vector storage for embedding documents and efficient retrieval.
- **Hugging Face**: Embedding models for document chunk embeddings.
- **OpenAI GPT-4**: Language model for question answering and summarization.
- **Document Loaders**: `PyPDFLoader` for PDFs, `Docx2txtLoader` for DOCX.

## Setup Instructions

### Prerequisites

1. **Python 3.7+**
2. **API Keys**:
   - OpenAI API Key for language generation (required for GPT-4).
   - Hugging Face API Key (if not already set up on your machine).
  
### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory to store your API keys.
   - Add the following lines, replacing with your actual keys:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     HF_TOKEN=your_huggingface_token
     ```

### Usage

1. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```
   
2. **Interacting with the App**:
   - **Upload Documents**: Choose files (PDF, DOCX, TXT, CSV) to upload.
   - **Ask Questions**: Type in questions related to the document content, and the app will retrieve relevant sections and generate concise answers.
   - **Request Summaries**: Use commands like `short summary` or `detailed summary` to specify the summary length.

## Project Structure

```
project-root/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not included in Git)
├── .gitignore            # Git ignore file to exclude sensitive files
└── README.md             # Project documentation
```

## Example Usage

- **Short Summary**: Enter “short summary” in the input box to get a high-level overview.
- **Detailed Summary**: Enter “detailed summary” for a more comprehensive breakdown.
- **Question**: Enter questions like “What is the main point in section 2?” to retrieve specific answers based on document context.

## Future Work

- **Integration with Additional Knowledge Sources**: Planned support for Wikipedia, arXiv, and web scraping for broader context.
- **Extended Language Support**: Potentially adding more languages or translation functionality.

## Contributing

Contributions are welcome! Please fork the repo and create a pull request with any improvements.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for language model prompt engineering.
- [Streamlit](https://streamlit.io/) for interactive data apps.
- [Chroma](https://www.trychroma.com/) for vector storage.

---

This README provides comprehensive setup instructions, feature descriptions, and usage guidance to help others understand and run your project on GitHub.

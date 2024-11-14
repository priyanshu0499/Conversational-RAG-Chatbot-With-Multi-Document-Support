import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

# Load API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
st.title("Conversational RAG with Document QA and Summarization")

if openai_api_key:
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
    if 'last_processed_input' not in st.session_state:
        st.session_state.last_processed_input = None
    if 'last_response' not in st.session_state:
        st.session_state.last_response = None

    uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt", "csv", "docx"], accept_multiple_files=True)
    documents = []

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_path = "./temp." + uploaded_file.name.split(".")[-1]
            with open(temp_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())

            elif uploaded_file.type == "text/plain":
                with open(temp_path, "r") as file:
                    content = file.read()
                documents.append(Document(page_content=content.strip(), metadata={"source": uploaded_file.name}))

            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                loader = Docx2txtLoader(temp_path)
                documents.extend(loader.load())

            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(temp_path)
                documents.append(Document(page_content=df.to_string(), metadata={"source": uploaded_file.name}))

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Set up history-aware retrieval and retrieval chain
        contextualize_q_system_prompt = (
    "You are tasked with making user questions self-contained by incorporating any relevant context from previous interactions. "
    "Rephrase the latest user question if needed to ensure it can stand alone without prior chat history. "
    "If the question is already clear, return it unchanged. Do not answer the question; only rephrase or clarify as needed."
)

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            ("human", "{input}")
        ])

        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        history_retrieval_chain = create_retrieval_chain(history_aware_retriever, retriever)

        # Prompts for QA and Summarization
        qa_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "You are an expert assistant, tasked with answering questions accurately and concisely based on relevant sections from the provided document(s). "
        "Focus on delivering precise information directly related to the question, avoiding any unrelated details. "
        "If the answer is unclear or not found in the documents, respond with: 'The document does not provide enough information to answer that question.'"
    ),
    ("human", "Question: {input}\nContext: {context}")
])

        summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "You are a summarization assistant. Generate a summary of the following content, adjusting the detail and length based on the user's preference. "
        "For short summaries, focus on main points only. For detailed summaries, provide a thorough overview of each significant topic covered."
    ),
    ("human", "{context}")
])

        def detect_summary_length(user_input):
            if "short" in user_input.lower():
                return "Provide a brief summary."
            elif "detailed" in user_input.lower():
                return "Provide a detailed summary."
            else:
                return "Provide a summary."

        def choose_prompt(user_input):
            return summarization_prompt if "summary" in user_input.lower() else qa_prompt

        with st.form(key='user_input_form', clear_on_submit=True):
            user_input = st.text_input("Your question or request for summary:", key='user_input')
            submit_button = st.form_submit_button("Submit")

        if submit_button and user_input:
            selected_prompt = choose_prompt(user_input)

            # Retrieve relevant sections based on user_input
            retrieved_documents = retriever.get_relevant_documents(user_input)
            retrieved_content = " ".join([doc.page_content for doc in retrieved_documents])

            # Adjust prompt for QA and Summary tasks
            if selected_prompt == qa_prompt:
                formatted_prompt = selected_prompt.format(input=user_input, context=retrieved_content)
            else:
                summary_length_instruction = detect_summary_length(user_input)
                formatted_prompt = selected_prompt.format(context=summary_length_instruction + "\n" + retrieved_content)

            # Generate response using the LLM
            response = llm(formatted_prompt)
            response_content = response.content if hasattr(response, 'content') else " ".join([msg.content for msg in response])

            # Update session state
            st.session_state.last_processed_input = user_input
            st.session_state.last_response = response_content

    # Display the latest response if available
    if st.session_state.last_response:
        st.markdown(f"**You:** {st.session_state.last_processed_input}")
        st.markdown(f"**Assistant:** {st.session_state.last_response}")

else:
    st.warning("Please ensure the OpenAI API key is set in the environment.")

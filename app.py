import gradio as gr  # Gradio is a library for creating web-based interfaces for machine learning models.
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader  # Loaders for reading data from the web and PDFs.
from langchain_community.vectorstores import Chroma  # Chroma is a vector store for embeddings.
from langchain_community import embeddings  # Embeddings module to generate vector representations.
from langchain_community.chat_models import ChatOllama  # ChatOllama is a local language model for conversational AI.
from langchain_core.runnables import RunnablePassthrough  # Allows data to be passed through unchanged.
from langchain_core.output_parsers import StrOutputParser  # Parses output to string.
from langchain_core.prompts import ChatPromptTemplate  # Template for creating prompt-based chats.
from langchain.output_parsers import PydanticOutputParser  # Parser for structured output.
from langchain.text_splitter import CharacterTextSplitter  # Splits text into smaller chunks.
import pandas as pd  # Pandas for handling CSV files.
from langchain.schema import Document  # Represents a text-based document.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text recursively into smaller chunks.
from ollama import Client  # Client to connect to the Ollama server.

# Define the client for the Ollama service on the local machine
client = Client(host='http://localhost:11434')

# Path to the test data, which is a CSV file
test_data = 'D:/Study/RAG/Book1.csv'

# Load the local chat model
model_local = ChatOllama(model="phi3")

# Function to load the CSV file, process it into documents, and create a vector store
def vector_input():
    # Load the CSV file into a DataFrame
    data = pd.read_csv(test_data)
    print(f"Loaded Excel file: {test_data}")  # Informational message for debugging/logging
    print(data.head(10))  # Display the first 10 rows for verification

    # List to store the document objects
    documents = []
    # Iterate over the rows in the DataFrame
    for index, row in data.iterrows():
        # Concatenate all column values into a single text block
        content = " ".join(map(str, row.values))
        # Create a Document object with the concatenated content and metadata
        document = Document(page_content=content, metadata={"row_index": index})
        # Add the Document to the list
        documents.append(document)

    # Ensure that the list contains only Document objects
    if not all(isinstance(doc, Document) for doc in documents):
        raise TypeError("Expected list of Document objects.")  # Raise an error if not

    # Use RecursiveCharacterTextSplitter to split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter()
    doc_splits = text_splitter.split_documents(documents)

    # Create embeddings for the split documents and store them in a Chroma vector store
    print("Starting embedding")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",  # Name of the collection in the vector store
        embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text')  # Embedding model to use
    )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    return retriever  # Return the retriever object

# Create a retriever from the vector_input function
retriever = vector_input()

# Function to process user input and run the RAG (Retrieval-Augmented Generation) process
def process_input(question):
    print("\n########\nAfter RAG\n")  # Debugging information to indicate the RAG process start

    # Template for generating the answer based on context and question
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    # Create a prompt using the template
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    # Chain of operations for RAG
    after_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # Pass context and question
            | after_rag_prompt  # Use the prompt template
            | model_local  # Apply the local model
            | StrOutputParser()  # Parse the output to a string
    )

    return after_rag_chain.invoke(question)  # Invoke the RAG chain with the user's question

# Define a Gradio interface for the process_input function
iface = gr.Interface(
    fn=process_input,  # Function to call when a user submits a question
    inputs=gr.Textbox(label="Question"),  # Input for the user to enter their question
    outputs="text",  # Output as plain text
    title="Document Query with Ollama",  # Title of the Gradio interface
    description="Question to query the documents."  # Description for the interface
)

iface.launch()  # Launch the Gradio interface to make it accessible via a web browser

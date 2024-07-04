import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import PyPDF2
import os
import io

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set up the app layout
st.set_page_config(page_title="Chat Your PDFs", page_icon="📄", layout="wide")
st.title("Chat Your PDFs 📄")  # Updated title with icon

# Initialize session state for chat history and PDF context
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_context' not in st.session_state:
    st.session_state.pdf_context = None

# Sidebar for file upload
with st.sidebar:
    st.header("Upload your PDF file")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], help="Upload your PDF file here to start the analysis.")
    if uploaded_file is not None:
        st.success("PDF File Uploaded Successfully!")
        
        # PDF Processing (using PyPDF2 directly)
        pdf_data = uploaded_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        pdf_pages = pdf_reader.pages

        # Create Context
        context = "\n\n".join(page.extract_text() for page in pdf_pages)

        # Split Texts
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        texts = text_splitter.split_text(context)

        # Chroma Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
        
        st.session_state.vector_index = vector_index  # Store in session state for later use
        st.session_state.pdf_context = context  # Store PDF context in session state

def get_response(question):
    vector_index = st.session_state.vector_index
    # Get Relevant Documents
    docs = vector_index.get_relevant_documents(question)

    # Define Prompt Template
    prompt_template = """
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context."
    Don't provide incorrect information.\n\n
    Context:\n {context}?\n
    Question:\n{question}\n
    Answer:
    """

    # Create Prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # Load QA Chain
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=google_api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # Get Response
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response['output_text']

# Main chat interface
st.subheader("Chat with your PDF")

if 'vector_index' in st.session_state:
    # Display Chat History
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for interaction in st.session_state.chat_history:
            st.write(f"**You:** {interaction['question']}")
            st.write(f"**Bot:** {interaction['answer']}")
            st.markdown("---")

    # User Question
    user_question = st.text_input("Ask a Question:", key="user_question", help="Type your question here and press Enter.", on_change=lambda: st.session_state.update({
        'question_submitted': True
    }))

    # If the user presses Enter
    if st.session_state.get('question_submitted'):
        # Get answer
        response = get_response(user_question)

        # Add interaction to chat history
        st.session_state.chat_history.append({"question": user_question, "answer": response})

        # Clear input field and reset question_submitted
        st.session_state.user_question = ""
        st.session_state.question_submitted = False
        st.experimental_rerun()
else:
    st.info("Please upload a PDF file to start chatting.")

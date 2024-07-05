Here is a draft of the README file for your repository:

---

# Gemini Pro RAG App

This repository contains a Streamlit application that implements Retrieval-Augmented Generation (RAG) using the Gemini Pro API. This app allows users to upload a PDF, and then ask questions about its content.

## Overview

The application processes a PDF by extracting its text, splitting it into manageable chunks, embedding these chunks into a vector space, and storing the embeddings in a vector database. When a user asks a question, the app retrieves the most relevant document chunks and generates a detailed answer using the Gemini Pro model.

![Flowchart](path/to/your/image1.png)
![Flowchart](path/to/your/image2.png)

## Architecture

1. **Load**: The PDF is loaded and its text is extracted using PyPDF2.
2. **Split**: The text is split into chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embed**: The text chunks are embedded into a vector space using GoogleGenerativeAIEmbeddings from the Gemini Pro API.
4. **Store**: The embeddings are stored in a vector database using Chroma.
5. **Retrieve**: When a user asks a question, the most relevant text chunks are retrieved from the vector database.
6. **Generate**: The retrieved chunks are used to generate a detailed answer using the Gemini Pro model.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/gemini-pro-rag-app.git
cd gemini-pro-rag-app
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

Create a `.env` file in the root directory of the project and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

To run the application, use the following command:

```bash
streamlit run app.py
```

## Application Structure

- **app.py**: Main application file. Contains the Streamlit code for the user interface and logic for handling user inputs and displaying responses.
- **config/globals.py**: Contains global configurations and initial prompt.
- **.env**: Environment variables file (not included in the repository, needs to be created by the user).

## Code Explanation

### Text Extraction

The function `extract_text_from_pdf` extracts text from an uploaded PDF file using PyPDF2:

```python
def extract_text_from_pdf(pdf):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf))
    text = "\n\n".join(page.extract_text() for page in pdf_reader.pages)
    return text
```

### Vector Index Initialization

The function `initialize_vector_index` initializes a vector index from the extracted text:

```python
def initialize_vector_index(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
    return vector_index
```

### Response Generation

The function `get_response` retrieves relevant documents and generates a response using the Gemini Pro model:

```python
def get_response(question):
    vector_index = st.session_state.vector_index
    docs = vector_index.get_relevant_documents(question)
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context."
    Don't provide incorrect information.\n\n
    Context:\n {context}?\n
    Question:\n{question}\n
    Answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=google_api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response['output_text']
```


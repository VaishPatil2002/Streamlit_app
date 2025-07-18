import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

#  Set your OpenRouter API key and base
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_TYPE"] = "open_ai"

#  Read file contents
def read_file(file):
    text = ""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    return text

#  Streamlit UI
st.title(" Chat with Your Documents (DeepSeek via OpenRouter)")

uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    full_text = ""
    for file in uploaded_files:
        file_text = read_file(file)
        full_text += file_text + "\n"

    st.success(" Files read and processed")

    #  Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([full_text])

    #  Embeddings using HuggingFace (no billing required)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # Use DeepSeek model from OpenRouter
    llm = ChatOpenAI(
        model_name="deepseek/deepseek-r1-0528:free",
        temperature=0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    #  Combine retriever + LLM
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    #  Chat interface
    st.subheader(" Ask something about your files")
    query = st.text_input("Your question:")
    query = st.text_input("Your question:", placeholder="e.g., What is the document about?")



    if query:
        with st.spinner(" Thinking..."):
            result = chain({"query": query})
            st.write("###  Answer:")
            st.write(result["result"])

            with st.expander(" Sources used"):
                for doc in result["source_documents"]:
                    st.markdown(doc.page_content)
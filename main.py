import streamlit as st
import faiss
print(faiss.__version__)
import faiss  # already installed as faiss-cpu
import sys
sys.modules["faiss"] = faiss

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil

# ✅ Ensure event loop exists (important for async Google client)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)


# -------------------- PDF Processing --------------------
def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""   # avoid NoneType error
    return text


def get_text_chunks(text):
    """Splits extracted text into chunks for embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    return chunks


# -------------------- Embeddings & Vector Store --------------------
embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings


def get_vector_store(text_chunks):
    """Creates and saves FAISS vector index from text chunks"""
    vector_store = FAISS.from_texts(text_chunks, embedding=get_embeddings())
    # Clear old index if exists
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    vector_store.save_local("faiss_index")


# -------------------- Conversational QA Chain --------------------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say:
    "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    """Handles user queries by retrieving docs and generating an answer"""
    if not os.path.exists("faiss_index"):
        st.error("⚠️ No FAISS index found. Please upload and process PDFs first.")
        return

    new_db = FAISS.load_local("faiss_index", get_embeddings(), allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    output = response.get("output_text") or response.get("result", "")
    st.write("### 📖 Reply:")
    st.write(output)


# -------------------- Streamlit UI --------------------
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("📑 Interactive RAG-based LLM for Multi-PDF Document Analysis", divider='rainbow')

    user_question = st.text_input("🔍 Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on 'Submit & Process'",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done! Now you can ask questions.")
            else:
                st.error("⚠️ Please upload at least one PDF.")


if __name__ == "__main__":
    main()
#streamlit run main.py
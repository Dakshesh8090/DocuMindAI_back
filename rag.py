import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import embeddings, index_name
import os
from langchain_community.vectorstores import Pinecone as PineconeVectorStore


async def process_pdf(file):

    session_id = str(uuid.uuid4())
    file_path = f"{session_id}.pdf"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name,
            namespace=session_id
        )

        print("Data stored successfully ")

    finally:
        if os.path.exists(file_path): 
            os.remove(file_path) 
            print("Temporary file deleted")

    return session_id

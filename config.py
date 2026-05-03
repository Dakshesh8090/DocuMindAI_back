import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import google.generativeai as genai 

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

index_name = "rag-index"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
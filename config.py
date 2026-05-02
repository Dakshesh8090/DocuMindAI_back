import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from google import genai

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

index_name = "rag-index"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
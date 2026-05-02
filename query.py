from config import embeddings, index_name, client
from google.genai import types
import os
from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

load_dotenv()

history = []
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

async def transform_query(question):
    history.append({
        "role": "user",
        "parts": [{"text": question}]
    })

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction="You are a query rewriting expert. Based on the provided chat history , rephrase the follow up user question into a complete, standalone question that can be understood without any chat history. Only output the rewritten question and nothing else. If the question is already standalone, just repeat the question as it is."),
    )

    history.pop()
    return response.text.strip()

    #convert the user query into vector 

vectorstore = PineconeVectorStore(
    index_name="rag-index",
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}
)

async def ask_question(session_id, question):

    queries = await transform_query(question)

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=session_id
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    docs = retriever.invoke(queries)

    print("\nRewritten Query:", queries)

    print("\nRetrieved Docs:")
    for doc in docs:
        print(doc.page_content[:200])

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer ONLY using the context.

    Context:
    {context}

    Question:
    {question}
    """


    history.append({
            "role": 'user',
            "parts": [{"text": queries}]
    })

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    history.append({
            "role": 'model',
            "parts": [{"text": response.text}]
        })
    
    print("\nAnswer:", response.text)

    return response.text
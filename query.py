import os
from dotenv import load_dotenv
import google.generativeai as genai
from config import embeddings, index_name
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


chat_histories = {}


def get_history(session_id):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]


def format_history(history):
    return "\n".join([
        f"{msg['role']}: {msg['text']}"
        for msg in history
    ])


def transform_query(session_id, question):
    history = get_history(session_id)

    history_text = format_history(history)

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
        You are a query rewriting expert.
        Rewrite the user question into a standalone question.

        Chat History:
        {history_text}

        User Question:
        {question}
        """

    response = model.generate_content(prompt)
    return response.text.strip()


def ask_question(session_id, question):

    history = get_history(session_id)

    # Step 1: rewrite query
    rewritten_query = transform_query(session_id, question)

    # Step 2: retrieve docs
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=session_id
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(rewritten_query)

    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 3: final answer
    prompt = f"""
        Answer ONLY using the context.

        Context:
        {context}

        Question:
        {rewritten_query}
        """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Step 4: store history properly
    history.append({"role": "user", "text": question})
    history.append({"role": "assistant", "text": response.text})

    return response.text
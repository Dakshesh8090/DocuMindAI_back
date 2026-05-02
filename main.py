from fastapi import FastAPI, UploadFile, File
from rag import process_pdf
from query import ask_question
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload(file : UploadFile = File(...)):
    session_id = await process_pdf(file) 
    return {"session_id": session_id}

@app.post("/query") 
async def query(data: dict): 
    answer = await ask_question(data["session_id"], data["question"]) 
    return {"answer": answer}
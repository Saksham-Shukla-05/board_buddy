from fastapi import FastAPI, Request
from rag_chain import initialize_rag_chain

app = FastAPI()
qa_chain = None  # lazy load

@app.get("/")
def root():
    return {"message": "BoardBuddy is alive!"}

@app.post("/ask")
async def ask_question(request: Request):
    global qa_chain

    data = await request.json()
    question = data.get("question")
    if not question:
        return {"error": "No question provided."}

    if qa_chain is None:
        print("🔁 Initializing chain on first request...")
        qa_chain = initialize_rag_chain()

    result = qa_chain.invoke({
        "question": question,
        "chat_history": []
    })

    return {"answer": result["answer"]}

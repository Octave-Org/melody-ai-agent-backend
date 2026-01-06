# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import pandas as pd
import io
# Import the new function name
from brain import init_melody_agent

app = FastAPI(title="Melody AI API")

# Simple storage for the agent (in memory)
active_agent = None

class ChatRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), api_key: str = Form(...)):
    global active_agent
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Initialize using the new safe function
        active_agent = init_melody_agent(df, api_key)
        
        return {"status": "success", "rows_loaded": len(df), "message": "Brain is ready!"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(request: ChatRequest):
    global active_agent
    if not active_agent:
        return {"error": "Brain not active. Please POST to /upload first."}
    
    try:
        response = active_agent.invoke({"input": request.question})
        return {"answer": response['output']}
    except Exception as e:
        return {"error": str(e)}
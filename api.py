# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import io
import os
from brain import init_melody_agent

app = FastAPI(title="Melody AI API")

# Simple storage for the agent (in memory)
active_agent = None

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Melody AI API is running! Go to /docs to test it."}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global active_agent
    
    try:
        # Get API key from environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            return {"error": "GOOGLE_API_KEY not configured on server. Please set it in Render environment variables."}
        
        # Process the file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Initialize Brain with the environment variable key
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

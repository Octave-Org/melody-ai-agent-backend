# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import pandas as pd
import io
import os # <--- Added this to read Render variables
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
async def upload_file(file: UploadFile = File(...), api_key: str = Form(None)): # <--- Changed to None (Optional)
    global active_agent
    try:
        # 1. Determine which key to use
        # If the user sent one, use it. Otherwise, look for the Render Environment Variable.
        final_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not final_api_key:
            return {"error": "No API key found. Please provide one or set GOOGLE_API_KEY on the server."}

        # 2. Process the file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # 3. Initialize Brain with the chosen key
        active_agent = init_melody_agent(df, final_api_key)
        
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

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import os
from brain import init_melody_agent, get_last_chart_base64

app = FastAPI(title="Melody AI API")

# --- 1. CORS CONFIGURATION (Crucial for Frontend) ---
origins = [
    "http://localhost:5173",      # Frontend Localhost
    "http://127.0.0.1:5173",
    "https://melody-ai-frontend.vercel.app" # Placeholder for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple storage for the agent (in memory)
active_agent = None

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Melody AI API is running! Go to /docs to test it."}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    api_key: str = Form(None)
):
    global active_agent
    try:
        # 1. Get Key
        final_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not final_key:
            return {"error": "No API Key found! Please set GOOGLE_API_KEY in Render."}

        # 2. Process File (Excel/TSV/CSV Support)
        content = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content))
        elif filename.endswith('.tsv'):
            df = pd.read_csv(io.BytesIO(content), sep='\t')
        else:
            # Default to CSV
            df = pd.read_csv(io.BytesIO(content))
        
        # 3. Initialize Brain
        active_agent = init_melody_agent(df, final_key)
        
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
        answer_text = response['output']
        
        # Check for Chart
        chart_data = get_last_chart_base64()
        
        result = {"answer": answer_text}
        if chart_data:
            result["image"] = f"data:image/png;base64,{chart_data}"
            
        return result
        
    except Exception as e:
        return {"error": str(e)}

# --- NEW ENDPOINTS (For Frontend Buttons) ---

@app.delete("/session")
async def delete_session():
    global active_agent
    active_agent = None
    return {"status": "success", "message": "Session cleared."}

@app.post("/rerun")
async def rerun_analysis():
    global active_agent
    if not active_agent:
        return {"error": "No active session."}
    # Clear memory but keep data
    active_agent.chat_history = []
    return {"status": "success", "message": "Memory cleared."}

@app.get("/export")
async def export_report():
    if not active_agent:
        return {"error": "No analysis to export."}
    return {
        "status": "partial_success", 
        "message": "PDF Export coming in v2.0."
    }

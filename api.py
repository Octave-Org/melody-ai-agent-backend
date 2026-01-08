from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware  # <--- NEW IMPORT
from pydantic import BaseModel
import pandas as pd
import io
import os
from brain import init_melody_agent, get_last_chart_base64

app = FastAPI(title="Melody AI API")

# --- 1. CORS CONFIGURATION (The Fix) ---
origins = [
    "http://localhost:5173",      # The Frontend Developer's Local Machine
    "http://127.0.0.1:5173",      # Alternative Localhost
    "https://your-frontend-url.vercel.app" # (Optional) Add their live URL here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Allow these specific origins
    allow_credentials=True,
    allow_methods=["*"],          # Allow all methods (POST, GET, DELETE)
    allow_headers=["*"],          # Allow all headers
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
        final_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not final_key:
            return {"error": "No API Key found! Please provide one or set GOOGLE_API_KEY on the server."}

        content = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content))
        elif filename.endswith('.tsv'):
            df = pd.read_csv(io.BytesIO(content), sep='\t')
        else:
            df = pd.read_csv(io.BytesIO(content))
        
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
        chart_data = get_last_chart_base64()
        
        result = {"answer": answer_text}
        if chart_data:
            result["image"] = f"data:image/png;base64,{chart_data}"
            
        return result
        
    except Exception as e:
        return {"error": str(e)}

# --- NEW ENDPOINTS REQUESTED (Placeholders for v2) ---

@app.delete("/session")
async def delete_session():
    """Clears the current active agent (Simulates deleting the report)."""
    global active_agent
    active_agent = None
    return {"status": "success", "message": "Session cleared. Please upload a new file."}

@app.post("/rerun")
async def rerun_analysis():
    """Clears conversation history but keeps the data."""
    global active_agent
    if not active_agent:
        return {"error": "No active session to re-run."}
    
    # Clear the chat history list in the micro-agent
    active_agent.chat_history = []
    return {"status": "success", "message": "Analysis reset. Agent memory cleared."}

@app.get("/export")
async def export_report():
    """
    Placeholder for PDF Export. 
    Frontend can hit this, but it just returns a message for now.
    """
    if not active_agent:
        return {"error": "No analysis to export."}
        
    return {
        "status": "partial_success", 
        "message": "Export feature coming in v2.0. Please screenshot the dashboard for now.",
        "data_summary": "Session is active."
    }

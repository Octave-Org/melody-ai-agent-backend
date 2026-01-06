# 🎵 Melody Insight Engine

**Scalable backend service orchestrating autonomous agents for automated data analysis using FastAPI and Google Gemini.**

Melody Insight Engine is the core intelligence layer for the Melody AI platform. It implements a ReAct-pattern agentic workflow to ingest structured datasets, execute statistical reasoning, and generate actionable business insights via a high-performance RESTful API.

---

## 🚀 Capabilities

### v1.0: Core Agentic Workflow (Live)

- **Autonomous Data Ingestion**  
  Parses and validates structured CSV datasets into memory-efficient DataFrames.

- **Natural Language Reasoning**  
  Translates plain English business queries (e.g., *What is the revenue trend?*) into executable Python logic.

- **Statistical Computation**  
  Performs aggregation, filtering, and descriptive statistics (mean, median, variance) via pandas tools.

- **Schema Recognition**  
  Automatically detects semantic types (categorical vs. numerical) to optimize analysis strategies.

- **Visualization Logic**  
  Generates Matplotlib chart objects programmatically based on user intent.

---

## 🔮 v2.0 Roadmap: Enterprise Intelligence

- [ ] Multi-Format Ingestion: Native support for Excel (.xlsx) and TSV formats  
- [ ] Visual Serialization: Returning rendered charts as Base64-encoded strings for frontend display  
- [ ] Automated Insight Loop: One-shot generation of comprehensive reports (Summary, Trends, Anomalies)  
- [ ] State Persistence: Database-backed session management for multi-user isolation and report history  
- [ ] Export Engine: PDF generation service for downloadable insight reports  

---

## 🛠️ Technology Stack

| Component        | Technology         | Description                                                   |
|------------------|--------------------|---------------------------------------------------------------|
| API Framework    | FastAPI            | Asynchronous Python web framework for high-throughput endpoints |
| LLM Kernel       | Google Gemini 2.5 Flash | Multimodal model powering the reasoning engine                |
| Orchestrator     | Custom MicroAgent  | Lightweight, dependency-free ReAct implementation             |
| Data Engine      | Pandas / NumPy     | Vectorized data manipulation and statistical analysis         |
| Visualization    | Matplotlib         | Programmatic plotting library                                 |
| Server           | Uvicorn            | ASGI server for production deployment                         |

---

## ⚡ API Documentation

### Base URL


---

### 1. Initialize Session (Upload)

Ingests the dataset and initializes the reasoning engine.

- **Endpoint:** `POST /upload`  
- **Content-Type:** `multipart/form-data`

**Parameters**

| Name     | Type    | Description                              |
|----------|---------|------------------------------------------|
| file     | Binary  | The CSV dataset to analyze               |
| api_key  | String  | Google Gemini API key for the session    |

**Example Response:**
```json
{
  "status": "success",
  "rows_loaded": 1500,
  "message": "Brain is ready! You can now ask questions."
}
```

### 2. Execute Query (Chat)
Submits a natural language query to the agent.

- **Endpoint:** `POST /chat`
- **Content-Type:** `application/json`

**Request Body:**
```json
{
  "question": "Which product category has the highest profit margin?"
}
```
**Example Response:**
```json

{
  "answer": "The 'Electronics' category has the highest profit margin at 24%, followed by 'Furniture' at 18%."
}
```

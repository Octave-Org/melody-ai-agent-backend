# brain.py
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool

# --- GLOBAL STORAGE ---
last_generated_figure = None

def get_last_figure():
    return last_generated_figure

# --- TOOLS ---
def get_data_info(df):
    cols = {
        "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    return json.dumps({"rows": len(df), "columns": len(df.columns), "details": cols}, indent=2)

def query_data_logic(df, query):
    try:
        q = query.lower()
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if "top" in q or "highest" in q:
            col = next((c for c in num_cols if c.lower() in q), num_cols[0] if len(num_cols)>0 else None)
            if col:
                n = 5 if "5" in q else 3
                return df.nlargest(n, col).to_json(orient="records")
        return "Could not find specific pattern. Please ask for 'top 3 [column]'."
    except Exception as e:
        return str(e)

def create_chart_logic(df, query):
    global last_generated_figure
    try:
        q = query.lower()
        fig, ax = plt.subplots(figsize=(10, 6))
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        chart_type = "none"
        if "pie" in q and len(cat_cols) > 0:
            data = df[cat_cols[0]].value_counts().head(5)
            ax.pie(data, labels=data.index, autopct='%1.1f%%')
            chart_type = "pie"
        elif "bar" in q and len(cat_cols) > 0:
            data = df[cat_cols[0]].value_counts().head(10)
            ax.barh(range(len(data)), data.values)
            chart_type = "bar"
        elif "hist" in q and len(num_cols) > 0:
            ax.hist(df[num_cols[0]], bins=20)
            chart_type = "histogram"
        else:
            return "Please specify pie, bar, or histogram chart."
            
        plt.tight_layout()
        last_generated_figure = fig
        return json.dumps({"status": "chart_created", "type": chart_type})
    except Exception as e:
        return f"Error: {e}"

def get_stats_logic(df):
    return df.describe().to_json()

def analyze_general_logic(df):
    return f"Dataset has {len(df)} rows. Key columns: {list(df.columns[:5])}"

# --- CUSTOM MICRO-AGENT (The Nuclear Fix) ---
# This replaces AgentExecutor and create_react_agent entirely.
class MicroAgent:
    def __init__(self, llm, tools, df):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.df = df
    
    def invoke(self, inputs):
        question = inputs['input']
        
        # 1. Construct the Prompt (ReAct Style)
        tool_desc = "\n".join([f"{t.name}: {t.description}" for t in self.tools.values()])
        tool_names = ", ".join(self.tools.keys())
        col_info = ", ".join(self.df.columns.astype(str).tolist()[:15])
        
        system_prompt = f"""You are Melody AI. Answer the following questions as best you can.
Dataset Columns: {col_info}

You have access to the following tools:
{tool_desc}

Use the following format strictly:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:"""

        # 2. Run the Loop (Manual ReAct Loop)
        max_steps = 5
        current_scratchpad = ""
        
        for step in range(max_steps):
            # Call Gemini
            response = self.llm.invoke(system_prompt + current_scratchpad).content
            current_scratchpad += response
            
            # Check for Final Answer
            if "Final Answer:" in response:
                final_ans = response.split("Final Answer:")[-1].strip()
                return {"output": final_ans}
            
            # Check for Action
            action_match = re.search(r"Action: (.*?)\nAction Input: (.*)", response, re.DOTALL)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()
                
                # Execute Tool
                if tool_name in self.tools:
                    try:
                        tool_result = self.tools[tool_name].run(tool_input)
                    except Exception as e:
                        tool_result = f"Error: {str(e)}"
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."
                
                # Append Observation
                observation = f"\nObservation: {tool_result}\nThought:"
                current_scratchpad += observation
            else:
                # If LLM didn't follow format, just return what it said
                return {"output": response}
                
        return {"output": "I ran out of thinking steps, but here is what I found: " + current_scratchpad[-100:]}

# --- INITIALIZER ---
def init_melody_agent(df, api_key):
    # 1. Define Tools
    tools = [
        Tool(name="GetDataInfo", func=lambda x: get_data_info(df), description="Get dataset structure"),
        Tool(name="QueryData", func=lambda x: query_data_logic(df, x), description="Get top values"),
        Tool(name="GetStatistics", func=lambda x: get_stats_logic(df), description="Get stats"),
        Tool(name="CreateChart", func=lambda x: create_chart_logic(df, x), description="Create a plot"),
        Tool(name="AnalyzeGeneral", func=lambda x: analyze_general_logic(df), description="General analysis")
    ]

    # 2. Connect to Brain (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

    # 3. Return our Custom Agent (No Imports Needed!)
    return MicroAgent(llm, tools, df)
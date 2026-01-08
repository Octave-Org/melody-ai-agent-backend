import pandas as pd
import numpy as np
import json
import matplotlib

# [CRITICAL FIX] Force Matplotlib to run in "Headless Mode" (No Window)
# This allows charts to be generated on servers/cloud (Codespaces, Docker, AWS)
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import io
import base64
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool

# --- GLOBAL STORAGE (For Charts) ---
last_generated_figure = None

def get_last_figure():
    return last_generated_figure

def get_last_chart_base64():
    """
    Encodes the last figure to Base64 for the API response.
    Includes memory cleanup to prevent server crashes.
    """
    global last_generated_figure
    if last_generated_figure is None:
        return None
    
    try:
        buf = io.BytesIO()
        # Save figure to buffer
        last_generated_figure.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        
        # [MEMORY FIX] Explicitly close the plot to free RAM
        plt.close(last_generated_figure)
        last_generated_figure = None
        return img_str
    except Exception as e:
        print(f"Error encoding chart: {e}")
        return None

# --- TOOLS ---
def get_data_info(df):
    """Get dataset structure"""
    cols = {
        "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    return json.dumps({
        "rows": len(df), 
        "columns": len(df.columns), 
        "details": cols
    }, indent=2)

def count_rows_logic(df, query=None):
    count = len(df)
    return json.dumps({"total_rows": count, "message": f"The dataset has {count} rows."})

def get_category_counts_logic(df, query):
    try:
        q = query.lower()
        all_cols = [str(c) for c in df.columns]
        
        # Smart column detection
        target_col = None
        if any(word in q for word in ["gender", "sex", "male", "female"]):
            target_col = next((c for c in all_cols if "gender" in c.lower() or "sex" in c.lower()), None)
        elif any(word in q for word in ["country", "countries", "nation"]):
            target_col = next((c for c in all_cols if "country" in c.lower()), None)
        elif any(word in q for word in ["status", "tier", "level", "membership"]):
            target_col = next((c for c in all_cols if any(w in c.lower() for w in ["status", "tier", "level", "membership"])), None)
        
        if not target_col:
            target_col = next((c for c in all_cols if c.lower() in q), None)
        
        if not target_col:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            return f"Could not identify column. Available: {', '.join(cat_cols[:5])}"

        counts = df[target_col].value_counts().head(50).to_dict()
        total_unique = int(df[target_col].nunique())
        
        return json.dumps({
            "column": target_col,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "total_unique": total_unique
        }, indent=2)

    except Exception as e:
        return f"Error: {str(e)}"

def query_data_logic(df, query):
    try:
        q = query.lower()
        num_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns]
        
        if any(w in q for w in ["top", "highest", "most", "expensive", "best"]):
            target_col = next((c for c in num_cols if c.lower() in q), None)
            
            if not target_col and num_cols:
                target_col = num_cols[0]
            
            if target_col:
                n = 5 if "5" in q else 3
                return df.nlargest(n, target_col).to_json(orient="records")
        
        return "Please specify pattern like 'top 3 by [column]'"
    except Exception as e:
        return f"Error: {str(e)}"

def create_chart_logic(df, query):
    """Create visualizations (Optimized for Backend)"""
    global last_generated_figure
    
    # Clear any previous plots to prevent overlapping
    plt.close('all') 
    
    try:
        q = query.lower()
        colors = ['#FF1B6D', '#8B1BA8', '#FF6B9D', '#B24BDB', '#FF9EC7']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        chart_type = "none"
        
        if any(w in q for w in ["top", "highest", "most", "expensive", "best"]) and ("bar" in q or "chart" in q):
            target_num = num_cols[0] if num_cols else None
            for c in num_cols:
                if c.lower() in q:
                    target_num = c
                    break
            
            if target_num:
                n = 5 if "5" in q else 10
                data = df.nlargest(n, target_num)
                labels = data[cat_cols[0]] if cat_cols else data.index
                
                y_pos = range(len(data))
                ax.barh(y_pos, data[target_num], color=colors[0])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()
                ax.set_xlabel(target_num)
                ax.set_title(f"Top {n} by {target_num}")
                chart_type = "bar_ranking"
        
        elif "pie" in q and len(cat_cols) > 0:
            data = df[cat_cols[0]].value_counts().head(5)
            ax.pie(data, labels=data.index, autopct='%1.1f%%', colors=colors)
            chart_type = "pie"
            
        elif "bar" in q and len(cat_cols) > 0:
            data = df[cat_cols[0]].value_counts().head(10)
            ax.barh(range(len(data)), data.values, color=colors[0])
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index)
            chart_type = "bar_count"
            
        elif "hist" in q and len(num_cols) > 0:
            target_col = num_cols[0]
            for c in num_cols:
                if c.lower() in q:
                    target_col = c
                    break
            ax.hist(df[target_col], bins=20, color=colors[0], edgecolor='white')
            chart_type = "histogram"
            ax.set_title(f"Distribution of {target_col}")
        
        else:
            plt.close(fig)
            return "Please specify: pie, bar, or histogram chart"
            
        plt.tight_layout()
        last_generated_figure = fig
        print(f"[CreateChart] Figure Generated: {chart_type}")
        return json.dumps({"status": "chart_created", "type": chart_type})
    except Exception as e:
        print(f"[CreateChart] Error: {e}")
        return f"Error: {e}"

def get_stats_logic(df):
    return df.describe().to_json()

def analyze_general_logic(df):
    return f"Dataset has {len(df)} rows and {len(df.columns)} columns. First 5 columns: {list(df.columns[:5])}"

# --- MICRO-AGENT ---
class MicroAgent:
    def __init__(self, llm, tools, df):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.df = df
        self.chat_history = []
        print(f"✓ Agent initialized with {len(df):,} rows")

    def _extract_text(self, content):
        if isinstance(content, str): return content
        if isinstance(content, dict): return content.get("text") or content.get("content") or str(content)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str): parts.append(item)
                elif isinstance(item, dict): parts.append(item.get("text", item.get("content", str(item))))
                else: parts.append(str(item))
            return "".join(parts)
        return str(content)

    def invoke(self, inputs):
        question = inputs['input']
        
        history_text = ""
        if self.chat_history:
            history_text = "PREVIOUS CONVERSATION:\n" + "\n".join(
                [f"User: {q}\nAI: {a}" for q, a in self.chat_history[-3:]]
            ) + "\n\n"

        tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        tool_names = ", ".join(self.tools.keys())
        
        col_list = self.df.columns.astype(str).tolist()
        col_info = ", ".join(col_list[:15])
        if len(col_list) > 15:
            col_info += f", ... ({len(col_list) - 15} more)"
        
        system_prompt = f"""You are Melody AI, a data analysis assistant.

DATASET INFO:
- Total Rows: {len(self.df):,}
- Total Columns: {len(self.df.columns)}
- Columns: {col_info}

{history_text}AVAILABLE TOOLS:
{tool_desc}

TOOL SELECTION RULES:
1. "how many rows/records/customers" → CountRows
2. "how many X" (where X is a category value) → GetCategoryCounts
3. "dataset info/structure" → GetDataInfo
4. "chart", "plot", "visualize" → CreateChart

FORMAT:
Question: [question]
Thought: [reasoning]
Action: [{tool_names}]
Action Input: [input]
Observation: [result]
... (repeat if needed)
Thought: I have the answer
Final Answer: [clear response]

Question: {question}
Thought:"""

        max_steps = 5
        scratchpad = ""
        
        for step in range(max_steps):
            msg = self.llm.invoke(system_prompt + scratchpad)
            response = self._extract_text(msg.content)
            scratchpad += response
            
            print(f"\n[Step {step+1}] Response:\n{response[:200]}...")
            
            if "Final Answer:" in response:
                final = response.split("Final Answer:")[-1].strip()
                self.chat_history.append((question, final))
                return {"output": final}
            
            action_match = re.search(r"Action:\s*(.*?)\s*\nAction Input:\s*(.*?)(?:\n|$)", response, re.DOTALL)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip().split('\n')[0].strip()
                
                print(f"[Action] {tool_name}('{tool_input}')")
                
                if tool_name in self.tools:
                    try:
                        tool_obj = self.tools[tool_name]
                        result = tool_obj.invoke(tool_input) if hasattr(tool_obj, "invoke") else tool_obj.func(tool_input)
                    except Exception as e:
                        result = f"Error: {str(e)}"
                else:
                    result = f"Error: Tool '{tool_name}' not found"
                
                scratchpad += f"\nObservation: {result}\nThought: "
            else:
                final = response.strip().split('\n')[-1].strip()
                self.chat_history.append((question, final))
                return {"output": final}
                
        return {"output": "Reached maximum steps. Please rephrase your question."}

# --- INITIALIZER ---
def init_melody_agent(df, api_key):
    print("\n" + "="*50)
    print("INITIALIZING MELODY AI BRAIN")
    print("="*50)
    
    if len(df) == 0: raise ValueError("DataFrame is empty!")

    # Define tools
    tools = [
        Tool(name="CountRows", func=lambda x: count_rows_logic(df, x), description="Returns TOTAL row count."),
        Tool(name="GetCategoryCounts", func=lambda x: get_category_counts_logic(df, x), description="Counts values in a column."),
        Tool(name="GetDataInfo", func=lambda x: get_data_info(df), description="Get dataset structure."),
        Tool(name="QueryData", func=lambda x: query_data_logic(df, x), description="Get top/filtered rows."),
        Tool(name="GetStatistics", func=lambda x: get_stats_logic(df), description="Get statistical summary."),
        Tool(name="CreateChart", func=lambda x: create_chart_logic(df, x), description="Create visualizations (bar, pie, histogram)."),
        Tool(name="AnalyzeGeneral", func=lambda x: analyze_general_logic(df), description="General dataset overview.")
    ]

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        google_api_key=api_key,
        temperature=0
    )
    
    print("✓ LLM initialized (gemini-3-flash-preview)")
    return MicroAgent(llm, tools, df)

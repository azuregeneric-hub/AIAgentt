import os
import pandas as pd
import sqlite3
import streamlit as st
from datetime import datetime
from typing import TypedDict, Union, Optional, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_community.utilities import SQLDatabase 
import plotly.express as px  # Import Plotly Express for interactive charts
import os
# ---------- CONFIGURATION ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 




     #os.getenv("OPENAI_API_KEY")  # Set this environment variable to your new key

# --- 1. CONFIGURATION ---

db_path = "my_data.db"
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- 2. DATABASE & KPI INITIALIZATION ---
def initialize_database():
    pl_csv = os.path.join(current_dir, "P&LL.csv")
    utt_csv = os.path.join(current_dir, "UTT.csv")
    kpi_excel = os.path.join(current_dir, "KPI_Definitions.xlsx")
    
    if os.path.exists(pl_csv) and os.path.exists(utt_csv):
        df_pl = pd.read_csv(pl_csv, low_memory=False)
        df_utt = pd.read_csv(utt_csv, low_memory=False)
        for df, col in [(df_pl, 'Month'), (df_utt, 'Date_a')]:
            if col in df.columns:
                df['Month'] = pd.to_datetime(df[col], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
                df.dropna(subset=['Month'], inplace=True)
        conn = sqlite3.connect(db_path)
        df_pl.to_sql("p_and_l", conn, if_exists="replace", index=False)
        df_utt.to_sql("utt", conn, if_exists="replace", index=False)
        conn.close()
        
        if os.path.exists(kpi_excel):
            # Load main KPI definitions (assuming first sheet)
            df_kpi = pd.read_excel(kpi_excel)
            kpi_content = df_kpi.to_string(index=False)
            
            # Load ColumnMapping sheet
            column_mapping_text = ""
            try:
                df_mapping = pd.read_excel(kpi_excel, sheet_name="ColumnMapping")
                mapping_str = df_mapping.to_string(index=False)
                column_mapping_text = f"""

=== AUTHORIZED COLUMN NAMES ===
Use ONLY these column names in queries. Always use the "Safe SQL Name" exactly as listed (including quotes if present).

{mapping_str}

Rules:
- For any column referenced in formulas, use the corresponding Safe SQL Name.
- Do not invent or modify column names.
- If a column is not listed, do not use it.
"""
            except ValueError:
                st.warning("ColumnMapping sheet not found. Using default column handling.")
            
            # Combine KPI definitions and column mapping
            full_context = kpi_content + column_mapping_text
            return full_context
    return None

kpi_context = initialize_database()
if kpi_context is None:
    st.error("Required files missing!")
    st.stop()

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# --- 3. STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    architect_plan: str      
    query: str               
    result: pd.DataFrame     
    chart_code: str          
    insight: str             
    error: Optional[str]
    schema: str
    is_greeting: bool

# --- 4. AGENT NODES ---
def architect_agent(state: AgentState):
    """
    Role: The Brain. Handles greetings and recursive KPI formula lookups.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Check for Greetings
    greetings = ["hi", "hello", "hey", "hii", "thanks", "thank you"]
    if state['question'].lower().strip() in greetings:
        return {
            "is_greeting": True, 
            "architect_plan": "GREETING", 
            "insight": "Hello! I am your Financial AI Intelligence partner. How can I help you analyze your metrics today?"
        }

    # Recursive KPI Lookup: Architect looks at CM%, then looks for Revenue/Cost definitions
    prompt = f"""
    You are the Architect Agent. Your job is to analyze the user question using the KPI context.
    
    KPI CONTEXT FROM EXCEL:
    {kpi_context}

    USER QUESTION: {state['question']}

    DIRECTIONS:
    1. Identify the primary KPI (e.g., CM%).
    2. Scrutinize the formula. If it refers to other metrics (e.g., Revenue, Cost), find those definitions too.
    3. List the EXACT 'Group1' categories needed for every part of the calculation.
    4. Specify if a JOIN between 'p_and_l' and 'utt' is necessary.
    5. Use ONLY column names from the === AUTHORIZED COLUMN NAMES === section, exactly as listed in "Safe SQL Name".
    6.FinalCustomerName also means account/accounts.So if ina query someone asks to calculate by account,that means groupby by FinalCustomerName
    """
    response = llm.invoke(prompt)
    return {"architect_plan": response.content, "is_greeting": False}

def sql_analyst_agent(state: AgentState):
    """
    Role: The Coder. Translates the Architect's plan into raw SQLite.
    """
    if state.get("is_greeting"):
        return {"query": "SKIP"}

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    today = datetime.now().strftime('%Y-%m-%d')
    
    prompt = f"""
    You are the Data Analyst Agent. Write a SQLite query based on this Architect Plan:
    {state['architect_plan']}

    SCHEMA: {state['schema']}
    JOIN RULES: 
    - p_and_l.Segment = utt.Segment
    - p_and_l.FinalCustomerName = utt.FinalCustomerName
    - p_and_l.Contract ID = utt.sales document
    - p_and_l.wbs id = utt.WBSID
    - p_and_l.Month = utt.Month
    - p_and_l.PVDG = utt.ParticipatingVDG
    - p_and_l.PVDU = utt.ParticipatingVDU
    - p_and_l."Exec DG" = utt.DeliveryGroup
    - p_and_l."Exec DU" = utt.Delivery_Unit
    
    CRITICAL RULES:
    - Use EXACT 'Group1' strings for filters.
 
    - Handle division by zero with NULLIF(denominator, 0).
    - Default grouping: FinalCustomerName.
    - Use ONLY column names from the === AUTHORIZED COLUMN NAMES === section, exactly as listed in "Safe SQL Name".
    - For percentage KPIs like CM%, always multiply the calculation by 100 to get values between 0 and 100 (not 0 to 1).
    - For filters on percentages (e.g., "less than 80"), treat as percentage (HAVING "CM%" < 80, not < 0.80).
    - FinalCustomerName also means account/accounts.So if ina query someone asks to calculate by account,that means groupby by FinalCustomerName
    - RETURN RAW SQL ONLY. No markdown blocks, no ```sql.
    """
    response = llm.invoke(prompt)
    # Cleaning Layer to prevent the "near ```sql" error
    clean_sql = response.content.replace("```sql", "").replace("```", "").strip()
    return {"query": clean_sql}

def execute_query_node(state: AgentState):
    if state.get("is_greeting") or state.get("query") == "SKIP": return {"result": pd.DataFrame()}
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(state["query"], conn)
        conn.close()
        return {"result": df, "error": None}
    except Exception as e:
        return {"error": str(e), "result": pd.DataFrame()} 

def visualizer_agent(state: AgentState):
    if state.get("is_greeting") or state.get("error") or state["result"].empty: return {"chart_code": ""}
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    cols = list(state['result'].columns)
    num_rows = len(state['result'])
    
    # We enforce cols[0] as X (Category) and cols[-1] as Y (Numeric)
    prompt = f"""
Generate Streamlit code to plot this data correctly using plotly.express as px.
DataFrame is 'df'.
Columns: {cols} (Note: {cols[0]} is the categorical dimension, {cols[-1]} is the numeric KPI calculation).
Number of rows: {num_rows}

Steps to follow:
1. Detect chart type:
   - If columns include 'Month' or date-like, use px.line(df, x='{cols[0]}', y='{cols[-1]}') for time series.
   - For all other categorical queries (like FinalCustomerName), ALWAYS use a vertical bar chart to keep categories on the X-axis: 
     fig = px.bar(df, x='{cols[0]}', y='{cols[-1]}', orientation='v')
2. Sort the df by the numeric column for better readability: df = df.sort_values('{cols[-1]}', ascending=False)
3. If the KPI name contains 'cm%' or 'percent', ensure the y-axis is formatted as a percentage and take the values of the calculated KPI to show in the y axis.
4. Update layout for axes:
   - fig.update_layout(
       xaxis_title='{cols[0].replace("_", " ").title()}',  
       yaxis_title='{cols[-1].replace("_", " ").title()}',  
       yaxis_tickformat='.2f%' if 'cm%' in '{cols[-1]}'.lower() or 'percent' in '{cols[-1]}'.lower() else '.2f',
       xaxis_showgrid=True, yaxis_showgrid=True,  
       xaxis_zeroline=True, yaxis_zeroline=True,  
       xaxis_tickangle=-45  # Always rotate X-axis labels to prevent overlap with many customers
   )
5. If time series, sort df by the date column: df = df.sort_values('{cols[0]}')
6. To handle hover details, add hover_data=['{cols[0]}', '{cols[-1]}'].
7. Finally, st.plotly_chart(fig, use_container_width=True)
8. Handle empty data: if df.empty, just st.write('No data to plot.')
RAW CODE ONLY. No explanations or markdown.
    """
    response = llm.invoke(prompt)
    return {"chart_code": response.content.strip().replace("```python", "").replace("```", "")}
def insights_agent(state: AgentState):
    if state.get("is_greeting"): return {"insight": state.get("insight")}
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    prompt = f"Provide 2-sentence business insight for: {state['result'].head().to_string()}"
    response = llm.invoke(prompt)
    return {"insight": response.content}

# --- 5. GRAPH ---
builder = StateGraph(AgentState)
builder.add_node("architect", architect_agent)
builder.add_node("sql_analyst", sql_analyst_agent)
builder.add_node("executor", execute_query_node)
builder.add_node("visualizer", visualizer_agent)
builder.add_node("insights", insights_agent)
builder.add_edge(START, "architect"); builder.add_edge("architect", "sql_analyst")
builder.add_edge("sql_analyst", "executor"); builder.add_edge("executor", "visualizer")
builder.add_edge("visualizer", "insights"); builder.add_edge("insights", END)
graph = builder.compile()

# --- 6. UI ---
st.set_page_config(page_title="Financial AI Agent", layout="wide")

st.markdown("""
    <style>
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th { text-align: center !important; }
    .stChatMessage { border-radius: 12px; border: 1px solid #dee2e6; background-color: white; }
    .main { background-color: #f8f9fa; }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Financial Multi-Agent Intelligence")

schema_info = db.get_table_info()
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], pd.DataFrame): st.dataframe(msg["content"], use_container_width=True)
        else: st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            output = graph.invoke({"question": prompt, "schema": schema_info})
            
            if output.get("error"):
                st.error(output["error"])
            else:
                raw_df = output.get("result", pd.DataFrame())
                
                # --- DATA CLEANING FOR DISPLAY & CHART ---
                if not raw_df.empty:
                    # Create display version
                    display_df = raw_df.copy()
                    for col in display_df.select_dtypes(include=['number']).columns:
                        if 'cm%' in col.lower() or 'percent' in col.lower():
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
                        else:
                            display_df[col] = display_df[col].round(2)

                    # --- UI LAYOUT ---
                    st.subheader("📊 Data Result")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    st.subheader("📈 Visualization")
                    try:
                        # 1. Ensure numeric columns are actually numeric (fix for "mixed types" error)
                        chart_df = raw_df.copy()
                        first_col = chart_df.columns[0]
                        last_col = chart_df.columns[-1]
                        # Force all columns except the first to be numeric, errors to NaN
                        for col in chart_df.columns[1:]:
                            chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
                        
                        # Drop NaNs if any
                        chart_df = chart_df.dropna(subset=[last_col])
                        
                        # Additional preprocessing: Sort by last column for better viz
                        chart_df = chart_df.sort_values(by=last_col)
                        
                        # If 'Month' exists, ensure it's datetime
                        if 'Month' in chart_df.columns:
                            chart_df['Month'] = pd.to_datetime(chart_df['Month'], errors='coerce')
                        
                        # 2. Execute chart code
                        if output.get("chart_code"):
                            # Log for debugging
                            with st.expander("Generated Chart Code (Debug)"):
                                st.code(output["chart_code"], language="python")
                            
                            # Exec in a safe locals dict to avoid polluting globals
                            local_vars = {'st': st, 'df': chart_df.copy(), 'px': px}  # Use copy to avoid modifying original
                            exec(output["chart_code"], globals(), local_vars)
                        else:
                            st.write("No visualization generated.")
                    except Exception as chart_err:
                        st.error(f"Chart generation failed: {chart_err}")
                        # Improved fallback: Use Plotly instead of Altair for consistency
                        if not raw_df.empty:
                            is_horizontal = len(chart_df) > 10
                            fig = px.bar(chart_df, y=first_col if is_horizontal else first_col, 
                                         x=last_col if is_horizontal else last_col, 
                                         orientation='h' if is_horizontal else 'v')
                            fig.update_layout(
                                xaxis_title=last_col.replace("_", " ").title() if is_horizontal else first_col.replace("_", " ").title(),
                                yaxis_title=first_col.replace("_", " ").title() if is_horizontal else last_col.replace("_", " ").title(),
                                yaxis_tickformat='.2f%' if '%' in last_col.lower() else '.2f',
                                xaxis_tickangle=-45 if not is_horizontal else 0
                            )
                            st.plotly_chart(fig, use_container_width=True)

                st.success(f"**Insight:** {output['insight']}")
                st.session_state.messages.append({"role": "assistant", "content": display_df if not raw_df.empty else "No data."})
                st.session_state.messages.append({"role": "assistant", "content": output["insight"]})

                with st.expander("Technical Log"):
                    st.code(output["query"], language="sql")
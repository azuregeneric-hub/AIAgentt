
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
import re  # Added for extracting potential filter values
 
# ---------- CONFIGURATION ----------
 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
# --- 1. CONFIGURATION ---
 
db_path = "my_data.db"
current_dir = os.path.dirname(os.path.abspath(__file__))
 
# --- 2. DATABASE & KPI INITIALIZATION ---
def initialize_database():
    pl_csv = os.path.join(current_dir, "P&Lold.csv")
    utt_csv = os.path.join(current_dir, "UTold.csv")
    kpi_excel = os.path.join(current_dir, "KPI_Definitions.xlsx")
   
    if os.path.exists(pl_csv) and os.path.exists(utt_csv):
        df_pl = pd.read_csv(pl_csv, low_memory=False)
        df_utt = pd.read_csv(utt_csv, low_memory=False)
        if 'Month' in df_pl.columns:
            df_pl['Month'] = pd.to_datetime(df_pl['Month'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
            df_pl.dropna(subset=['Month'], inplace=True)
       
        # Normalize UTT 'Date_a' in-place (no new 'Month' column)
        if 'Date_a' in df_utt.columns:
            df_utt['Date_a'] = pd.to_datetime(df_utt['Date_a'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
            df_utt.dropna(subset=['Date_a'], inplace=True)
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
 
 
from difflib import get_close_matches  # For fuzzy
from collections import defaultdict
 
# Expanded candidates (add ProfitCentre if exists)
candidate_columns = {
    "utt": ["WBSID", "PSNo", "sales document", "FinalCustomerName", "ProfitCentre", "BusinessUnit", "Segment", "DeliveryGroup", "Delivery_Unit"],
    "p_and_l": ["wbs id", "Contract ID", "FinalCustomerName", "Segment"]
}
 
value_to_cols = defaultdict(list)  # {val: [(table, col, count), ...]}
 
# In build_reverse_lookup function (change the length filter)
def build_reverse_lookup():
    global value_to_cols
    conn = sqlite3.connect(db_path)
    value_to_cols.clear()
    for table, cols in candidate_columns.items():
        for col in cols:
            try:
                # Get distinct values + counts
                query = f'SELECT "{col}", COUNT(*) as cnt FROM "{table}" WHERE "{col}" IS NOT NULL AND "{col}" != "" GROUP BY "{col}" HAVING cnt > 0'
                df = pd.read_sql_query(query, conn)
                for _, row in df.iterrows():
                    val = str(row[col]).strip().upper()  # Normalize: upper, no spaces/punct
                    if len(val) >= 2:  # Changed to >=2 to include short IDs like 'A1'
                        value_to_cols[val].append((table, col, int(row['cnt'])))
            except Exception as e:
                print(f"Skip {table}.{col}: {e}")
    conn.close()
    # Sort each by count DESC (best first)
    for val in value_to_cols:
        value_to_cols[val].sort(key=lambda x: x[2], reverse=True)
    print(f"Built reverse lookup: {len(value_to_cols)} keys")
 
build_reverse_lookup()  # Run once
 
 
 
 
# --- 3. STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    architect_plan: str      
    disambiguated_plan: str  # New: Updated plan after column disambiguation
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
    Role: The Brain. Handles greetings, chit-chat, and recursive KPI formula lookups.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
   
    # Use LLM to classify intent
    intent_prompt = f"""
    Classify the user question: "{state['question']}"
   
    Possible intents:
    - GREETING_CHITCHAT: Greetings (hi, hello, hey, variations/misspellings), casual questions (how are you, what do you do, what's your work, who are you), off-topic (weather today, unrelated to finance).
    - KPI_QUERY: Questions about financial metrics, KPIs, data analysis from the database.
   
    If GREETING_CHITCHAT, output only "GREETING_CHITCHAT".
    If KPI_QUERY, output only "KPI_QUERY".
    If unsure, default to KPI_QUERY.
    """
    intent_response = llm.invoke(intent_prompt)
    intent = intent_response.content.strip()
   
    if intent == "GREETING_CHITCHAT":
        # Generate engaging response
        today = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        greeting_prompt = f"""
        User query: "{state['question']}"
        Current time - {today}
       
        Generate an engaging, active response as a Financial AI Intelligence partner.
        Personalize with  time-based greeting as per IST, and make it fun/professional.
        For off-topic like weather, politely redirect: "I'm focused on financial analysis, but if you meant something else, let me know!"
        For "what do you do": Explain role briefly.
        Keep concise, end with a question like "What financial metrics can I help with today?"
        Examples:
        - "how is the weather today": "Hey ! I'm not a weather bot, but here in Dehrdaun it's probably sunny. 😊 As your Financial AI, how can I assist with metrics?"
        - "what do u do": "Hi ! I analyze financial KPIs like CM%, Utilization, and more from your data. What's a metric you'd like insights on?"
        """
        response = llm.invoke(greeting_prompt)
        return {
            "is_greeting": True,
            "architect_plan": "GREETING",
            "insight": response.content
        }
 
    # If KPI_QUERY, proceed as before
    today = datetime.now().strftime('%Y-%m-%d')
 
    # Recursive KPI Lookup: Architect looks at CM%, then looks for Revenue/Cost definitions
    prompt = f"""
    You are the Architect Agent. Your job is to analyze the user question using the KPI context.
   
    Current date: {today}
   
    KPI CONTEXT FROM EXCEL:
    {kpi_context}
 
    USER QUESTION: {state['question']}
 
    DIRECTIONS:
    1. Identify the primary KPI (e.g., CM%, Utilization,UT).
    2. Scrutinize the formula. If it refers to other metrics (e.g., Revenue, Cost,NetAvailableHours,TotalBillableHours etc), find those definitions too.
    3. List the EXACT 'Group1' categories needed for every part of the calculation.
    4. Specify if a JOIN between 'p_and_l' and 'utt' is necessary.
    5. Use ONLY column names from the === AUTHORIZED COLUMN NAMES === section, exactly as listed in "Safe SQL Name".
    6. FinalCustomerName also means account/accounts.So if in a query someone asks to calculate by account,that means groupby by FinalCustomerName
    7.If in a KPI calculation, there is a division, so plz mutilpy that with 100 in order to get percentage value.
    8. For KPI calculations that come from the utt table, use the 'Date_a' field (normalized to YYYY-MM-DD) for time dimensions and to apply time filters.
    9. If the query references specific values (e.g., 'A1') without specifying the column, note them for potential column disambiguation in filters. Suggest likely columns based on context (e.g., if 'for account A1', note 'likely FinalCustomerName'). Do NOT add the filter in the plan—let disambiguator resolve. If value is a number like '30', treat as threshold, not filter value.
    10.DU also means 'Exec DU' column from p_and_l or  'Delivery_Unit' column from utt table .So if in a query someone asks to calculate/display by DU,that means groupby by 'Exec DU' column from p_and_lor by 'Delivery_Unit' from utt depending on the query
    11.BU also means 'Exec DG' column from p_and_l or 'DeliveryGroup' column from utt table .So if in a query someone asks to calculate/display by BU,that means groupby by 'Exec DG' column from p_and_l or by 'DeliveryGroup' from utt depending on the query
    12. If the user question contains relative time references like "this quarter", "last quarter", "YTD", etc., resolve them to specific date ranges based on the current date.
    - Quarters are defined as: Q1: Jan 1 - Mar 31, Q2: Apr 1 - Jun 30, Q3: Jul 1 - Sep 30, Q4: Oct 1 - Dec 31.
    - "This quarter" is the quarter containing the current date.
    - "Last quarter" is the previous quarter; if current is Q1, last is Q4 of previous year.
    - Specify the exact BETWEEN 'start_date' AND 'end_date' for each in the plan (e.g., this quarter: '2026-01-01' AND '2026-03-31').
    13. If the query mentions multiple grouping fields like 'DU/BU/account', interpret it as requiring grouping by each field separately.  For trends, include a time dimension like quarter or month in the grouping as meantioned in the user query.Dont show the time dimension in user table.
    14. If the query mentions by specific month, please use the correct date format of '%Y-%m-%d' , as also defined earlier.
    15. Do not add filters for specific values in the plan unless the query explicitly specifies the column (e.g., "for account A1"). For ambiguous "for A1", let the disambiguator resolve and add the filter.
    16.Since the Month column from p_and_l and Date_a column from utt contains only the starting date of each month, so when asked for  a specific month filter, dont apply the last date of the month in between filter.Also applu 2025 year only if the query does not mention any year
    """
    response = llm.invoke(prompt)
    return {"architect_plan": response.content, "is_greeting": False}
 
# In disambiguator_agent function (change the regex)
# In disambiguator_agent function (refine regex to require starting with a letter)
# In disambiguator_agent (disable fuzzy for short IDs if no exact match, to avoid wrong columns)
def disambiguator_agent(state: AgentState):
    if state.get("is_greeting"):
        return {"disambiguated_plan": state["architect_plan"]}

    potential_values = re.findall(r'\b[A-Z][A-Z0-9-]{1,}\b', state['question'])
   
    if not potential_values:
        return {"disambiguated_plan": state["architect_plan"]}
   
    filters = []
    is_account_query = any(word in state['question'].lower() for word in ['account', 'customer', 'finalcustomername'])  # Enhanced context check
    for val in set(pv.upper() for pv in potential_values):
        if val in value_to_cols and value_to_cols[val]:
            candidates = value_to_cols[val]
            # Prefer 'FinalCustomerName' if in candidates and query suggests account
            preferred_candidates = [c for c in candidates if c[1] == 'FinalCustomerName'] if is_account_query else candidates
            best = preferred_candidates[0] if preferred_candidates else candidates[0]
            table_alias = 'u' if 'utt' in best[0] else 'p'
            filter_sql = f'{table_alias}.\"{best[1]}\" = \'{val}\''
            filters.append(filter_sql)
            print(f"Matched '{val}' → {best[0]}.\"{best[1]}\" ({best[2]} rows)")
        else:
            # Skip fuzzy for short values (<5 chars) to avoid mismatches; only use if exact not found but longer
            if len(val) >= 5:
                fuzzy = []
                for v in value_to_cols:
                    if len(get_close_matches(val, [v], n=1, cutoff=0.8)):
                        fuzzy.append(value_to_cols[v][0])
                if fuzzy:
                    best = max(fuzzy, key=lambda x: x[2])
                    filters.append(f'{ "u" if "utt" in best[0] else "p" }.\"{best[1]}\" LIKE \'%{val}%\'')
            else:
                print(f"No exact match for short value '{val}'; skipping fuzzy to avoid errors.")

    if filters:
        filter_str = " AND " + " AND ".join(filters)
        updated_plan = state["architect_plan"] + f"\n\nADD FILTERS: {filter_str}"
    else:
        updated_plan = state["architect_plan"] + "\n\nNo matching columns for filters."
   
    return {"disambiguated_plan": updated_plan}
 
 
def sql_analyst_agent(state: AgentState):
    """
    Role: The Coder. Translates the Architect's plan into raw SQLite.
    """
    if state.get("is_greeting"):
        return {"query": "SKIP"}
 
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    today = datetime.now().strftime('%Y-%m-%d')
   
    prompt = f"""
    You are the Data Analyst Agent. Write a SQLite query based on this Disambiguated Plan:
    {state['disambiguated_plan']}
 
    Current date: {today}
   
    SCHEMA: {state['schema']}
    JOIN RULES:
    - p_and_l.Segment = utt.Segment
    - p_and_l.FinalCustomerName = utt.FinalCustomerName
    - p_and_l.Contract ID = utt.sales document
    - p_and_l.wbs id = utt.WBSID
    - p_and_l.Month = utt.Date_a
    - p_and_l.PVDG = utt.ParticipatingVDG
    - p_and_l.PVDU = utt.ParticipatingVDU
    - p_and_l.Exec DG = utt.DeliveryGroup
    - p_and_l.Exec DU = utt.Delivery_Unit
   
    CRITICAL RULES:
    - Use EXACT 'Group1' strings for filters.
    - Use Exact column names for joining the two tables.Do not add any join conditions beyond the exact JOIN RULES provided.
    - Handle division by zero with NULLIF(denominator, 0).
    - Default grouping: FinalCustomerName.
    - Use ONLY column names from the === AUTHORIZED COLUMN NAMES === section, exactly as listed in "Safe SQL Name".
    - For percentage KPIs like CM%,UT,Billed rate,Realized Rate,Revenue Per Person,Onsite Utilization,Offshore Utilization always multiply the calculation by 100 to get values between 0 and 100 (not 0 to 1).
    - For filters on percentages (e.g., "less than 80"), treat as percentage (HAVING "CM%" < 80, not < 0.80).
    - FinalCustomerName also means account/accounts.So if ina query someone asks to calculate by account,that means groupby by FinalCustomerName.
    - For time-based filters or grouping (e.g., monthly), use p.Month or u.Date_a, as they are both normalized to 'YYYY-MM-DD' format (e.g., '2025-12-01').
    - When joining p_and_l and utt, ALWAYS include ALL the listed JOIN RULES exactly as specified in the ON clause, regardless of whether the calculation is aggregate/overall or grouped (e.g., per FinalCustomerName). This ensures accurate matching of records before any summation or grouping to avoid data inflation or mismatches.
    - For time filters on utt, use only the join on p.Month = u.Date_a and the filter on p.Month. Do NOT add redundant BETWEEN clauses on u.Date_a, as Date_a represents the 1st of the month and the join handles the matching.
    - Always GROUP BY p.FinalCustomerName (not u.FinalCustomerName) for consistency, as p_and_l is the primary revenue source.
    - If the plan specifies relative time references, use the current date to confirm date ranges, but prefer the resolved ranges from the plan.
    - If the plan requires grouping by multiple fields separately (e.g., for trends by DU, BU, account), create a query that computes the KPI for each grouping separately, perhaps using UNION ALL with an additional column indicating the grouping type (e.g., 'Group Type' as 'DU', 'BU', or 'Account'), and include the time dimension (e.g., quarter) in the GROUP BY.
    - For Margin drop calculation, For column "Segment" equals to 'Transportation', Group the column "Group Description" by sum of values in column "Amount in USD" where Column "Type" is 'Cost'
      Check the difference of these grouped costs between last month and its previous month.List Costs which have increased in last month as compared to its previous month.Do not apply  filters on Month in/Month between , "Group1" , "Exec DU". Also 'transportation' from "Segment" column is in lower case.Dont apply any  additional filter on Month. Apply "or" condition between Type and Segment filter.
    -If the user asks for rate drops, increases, or comparisons (e.g., “realized rate dropped more than $3”), generate a CTE that computes the KPI for both the current and previous period (typically month or quarter), and then compare them in the final SELECT. Always compute both values explicitly — do not reference non-existent columns like PreviousRealizedRate.
    -Apply ONLY the exact ADD FILTERS from the disambiguated plan for specific values. If no ADD FILTERS, do NOT add any for mentioned values. For thresholds like '< 30', use in HAVING, not as string filters (e.g., no LIKE '%30%').
    - For aliases in SQL, replace special characters like '&' with '_and_' (e.g., 'C&B' becomes 'C_and_B') to avoid syntax errors. Do not use unquoted aliases with special characters.
    🚨 SQLITE AGGREGATE RULE (MANDATORY):
    - WHERE = row-level filters (Month, WBSID, FinalCustomerName)
    - HAVING = aggregate filters (SUM, COUNT, CM% < 30, avg > 80%)
    - ORDER: WHERE → GROUP BY → HAVING → ORDER BY
    Examples:
    ❌ WRONG: WHERE SUM(revenue) > 1000
    ✅ RIGHT: GROUP BY customer HAVING SUM(revenue) > 1000
    ❌ WRONG: WHERE CM% < 30
    ✅ RIGHT: GROUP BY FinalCustomerName HAVING CM% < 30
 
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
builder.add_node("disambiguator", disambiguator_agent)  # New node
builder.add_node("sql_analyst", sql_analyst_agent)
builder.add_node("executor", execute_query_node)
builder.add_node("visualizer", visualizer_agent)
builder.add_node("insights", insights_agent)
builder.add_edge(START, "architect")
builder.add_edge("architect", "disambiguator")  # New edge
builder.add_edge("disambiguator", "sql_analyst")  # New edge
builder.add_edge("sql_analyst", "executor")
builder.add_edge("executor", "visualizer")
builder.add_edge("visualizer", "insights")
builder.add_edge("insights", END)
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
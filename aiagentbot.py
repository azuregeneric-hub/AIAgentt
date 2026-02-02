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
from langgraph.checkpoint.memory import MemorySaver
import uuid
import pickle




DB_FILE = "financial_agent_history.pkl"

def save_history():
    with open(DB_FILE, "wb") as f:
        pickle.dump(st.session_state.all_chats, f)

def load_history():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                return pickle.load(f)
        except:
            return {}
    return {}

def clear_all_history_file():
    """Permanently deletes the local pickle file and resets state."""
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    st.session_state.all_chats = {}
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
def delete_chat_session(tid):
    """Deletes a specific chat from the registry and disk."""
    if tid in st.session_state.all_chats:
        del st.session_state.all_chats[tid]
        save_history()
        
        # If we just deleted the chat we are currently looking at, reset to a new one
        if st.session_state.thread_id == tid:
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
# --- INITIALIZE STATE ---
if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_history()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

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

    history: List[str]

    architect_plan: str      
    disambiguated_plan: str  
    query: str               
    result: pd.DataFrame    
    chart_code: str          
    insight: str            
    error: Optional[str]
    schema: str
    is_greeting: bool
    is_explanation: bool
    # --- New Fields for Conversation ---
    needs_clarification: bool
    clarification_options: List[str]
 
# --- 4. AGENT NODES ---
def architect_agent(state: AgentState):
    """
    Role: The Brain. Handles greetings, chit-chat, and recursive KPI formula lookups.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    history_context = "\n".join(state.get("history", []))
    
    # Use LLM to classify intent
    intent_prompt = f"""
    Classify the user question: "{state['question']}"
    CONVERSATION HISTORY:
    {history_context}
    Possible intents:
    - GREETING_CHITCHAT: Greetings (hi, hello, hey, variations/misspellings), casual questions (how are you, what do you do, what's your work, who are you), off-topic (weather today, unrelated to finance).
    - KPI_EXPLANATION: Questions asking for definitions, formulas, components (e.g., numerator, denominator), how a KPI is calculated, or breakdowns of metrics (e.g., "what is CM%?", "numerator for Revenue?", "explain the formula for Cost"). If the question uses words like 'give the formula', 'what is the formula', 'explain the formula', or similar, classify as KPI_EXPLANATION.
    - KPI_QUERY: Questions about computing or analyzing financial metrics, KPIs, data from the database (e.g., "calculate CM% for last quarter", "show Utilization by account").
   
    Examples:
    - "hi there": GREETING_CHITCHAT
    - "what is the formula for CM%?": KPI_EXPLANATION
    - "give the formula for cm%": KPI_EXPLANATION
    - "give the numerator and denominator for utilization": KPI_EXPLANATION
    - "calculate cm% for transportation": KPI_QUERY
    - "show formula and calculate cm%": KPI_QUERY  # Mixed, but involves computation, so QUERY
   
    Output ONLY the intent name (e.g., "KPI_EXPLANATION"). If unsure, default to KPI_QUERY.
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
            "is_explanation": False,
            "architect_plan": "GREETING",
            "insight": response.content
        }
    
    elif intent == "KPI_EXPLANATION":
       today = datetime.now().strftime('%Y-%m-%d')
       explanation_prompt = f"""
       You are a helpful Financial AI. Provide a clear, natural language explanation of the KPI based on this context.

       KPI CONTEXT FROM EXCEL:
       {kpi_context}

       USER QUESTION: {state['question']}
       Current date: {today}

       DIRECTIONS:
       1. Identify the KPI (e.g., CM%, Revenue, Cost).
       2. Extract and explain the formula in simple terms.
       3. For questions about numerator/denominator: Break it down explicitly (e.g., "Numerator: Revenue - Cost; Denominator: Revenue").
       4. Use bullet points for clarity (e.g., - Numerator: ..., - Denominator: ...).
       5. Make it conversational: Start with "Sure, let me explain..." and end with "Does that help, or do you want an example calculation?"
       6. If the formula references other KPIs, recursively explain them briefly (e.g., "CM% is (Revenue - Cost) / Revenue * 100. Revenue is the sum of...").
       7. Reference exact 'Group1' categories from the context.
       8. No SQL or data queries—just explanation.

       Keep response concise and natural.
       """
       response = llm.invoke(explanation_prompt)
       return {
        "is_greeting": False,
        "is_explanation": True,
        "architect_plan": "EXPLANATION",
        "insight": response.content
       }

    else:  # KPI_QUERY (explicit else for clarity)
        today = datetime.now().strftime('%Y-%m-%d')
 
        # Recursive KPI Lookup: Architect looks at CM%, then looks for Revenue/Cost definitions
        prompt = f"""
        You are the Architect Agent. Your job is to analyze the user question using the KPI context and conversation history.
       
        Current date: {today}
       
        KPI CONTEXT FROM EXCEL:
        {kpi_context}

        CONVERSATION HISTORY (Use this to carry over filters):
        {history_context}

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
        16.Since the Month column from p_and_l and Date_a column from utt contains only the starting date of each month, so when asked for  a specific month filter, dont apply the last date of the month in between filter.Also applu 2025 year only if the query does not mention any year.
        17.CM also means CM%.
        18. **Root Cause / Factor Analysis**: If the user asks for "factors," "why," or "reasons" for a value (e.g., "Why is cost high for A24?"):
        - Do NOT generate a plan for a single total.
        - Instead, create a plan to BREAK DOWN the metric by "Group Description" and "Group1".
        - If a specific customer is mentioned (e.g., 'A24'), the plan must specify: "Filter by FinalCustomerName = 'A24' and GROUP BY 'Group Description'".
        - This allows the user to see which specific expenses (like Fuel, Maintenance, etc.) contributed most to the total.
        19. **MEMORY & FOLLOW-UPS**: If the user question is a follow-up (e.g., "Why?", "Factors for A42?", "What about the second one?"), 
           you MUST carry over filters (Month, Segment, DU, BU) from the CONVERSATION HISTORY.
           - Example: If the previous query was for "Jan 2025", and this question is "Why is A42 high?", 
             the plan MUST include "Filter: Month = '2025-01-01'".
        STRICT HIERARCHY RULES:
        1. **Formula Lookup**: If a user asks for "Cost", look at the 'Formula' column in the KPI context. 
        For Cost, you MUST use: "Group1" IN ('Direct Expense', 'OWN OVERHEADS', 'Indirect Expense', 'Project Level Depreciation', 'Direct Expense - DU Block Seats Allocation','Direct Expense - DU Pool Allocation','Establishment Expenses')
        2. **Segment Filtering**: If the question mentions "Transportation", "Logistics", etc., these are values in the "Segment" column. 
        3. **Combined Logic**: For "Transportation Cost", the plan must include:
        - Filter by "Segment" = 'Transportation'
        - Filter by the list of "Group1" categories found in the Cost formula.
   
        4. **Join Logic**: If the calculation requires both P&L and UTT, specify the join. Otherwise, use 'p_and_l' as the primary table for financial costs.
        5. **Avoid Hallucination**: Do NOT use "Type" = 'Cost' unless specifically asked. Use the 'Group1' categories for all KPI math.
        """
        response = llm.invoke(prompt)
        return {"architect_plan": response.content, "is_greeting": False, "is_explanation": False}

# ... (disambiguator_agent remains unchanged)

def sql_analyst_agent(state: AgentState):
    """
    Role: The Coder. Translates the Architect's plan into raw SQLite.
    """
    if state.get("is_greeting") or state.get("is_explanation"):
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
    - If the plan is purely explanatory (e.g., describes a formula without requesting computation, grouping, or filters), output 'SKIP' — do not generate SQL.
    - Use EXACT 'Group1' strings for filters.
    - Use Exact column names for joining the two tables.Do not add any join conditions beyond the exact JOIN RULES provided.
    - Handle division by zero with NULLIF(denominator, 0).
    - Default grouping: FinalCustomerName.
    - Use ONLY column names from the === AUTHORIZED COLUMN NAMES === section, exactly as listed in "Safe SQL Name".
    - For percentage KPIs like CM%,UT,Billed rate,Realized Rate,Revenue Per Person,Onsite Utilization,Offshore Utilization always multiply the calculation by 100 to get values between 0 and 100 (not 0 to 1).
    - For filters on percentages (e.g., "less than 80"), treat as percentage (HAVING "CM%" < 80, not < 0.80).
    - FinalCustomerName also means account/accounts.So if in a query someone asks to calculate by account,that means groupby by FinalCustomerName.
    - For time-based filters or grouping (e.g., monthly), use p.Month or u.Date_a, as they are both normalized to 'YYYY-MM-DD' format (e.g., '2025-12-01').
    - When joining p_and_l and utt, ALWAYS include ALL the listed JOIN RULES exactly as specified in the ON clause, regardless of whether the calculation is aggregate/overall or grouped (e.g., per FinalCustomerName). This ensures accurate matching of records before any summation or grouping to avoid data inflation or mismatches.
    - For time filters on utt, use only the join on p.Month = u.Date_a and the filter on p.Month. Do NOT add redundant BETWEEN clauses on u.Date_a, as Date_a represents the 1st of the month and the join handles the matching.
    - Always GROUP BY p.FinalCustomerName (not u.FinalCustomerName) for consistency, as p_and_l is the primary revenue source.
    - If the plan specifies relative time references, use the current date to confirm date ranges, but prefer the resolved ranges from the plan.
    - If the plan requires grouping by multiple fields separately (e.g., for trends by DU, BU, account), create a query that computes the KPI for each grouping separately, perhaps using UNION ALL with an additional column indicating the grouping type (e.g., 'Group Type' as 'DU', 'BU', or 'Account'), and include the time dimension (e.g., quarter) in the GROUP BY.
    - For Margin drop calculation, For column "Segment" equals to 'Transportation', Group the column "Group Description" by sum of values in column "Amount in USD" where Column "Type" is 'Cost'
      Check the difference of these grouped costs between last month and its previous month.List Costs which have increased in last month as compared to its previous month.Do not apply  filters on Month in/Month between , "Group1" , "Exec DU". Also 'transportation' from "Segment" column is in lower case.Dont apply any  additional filter on Month. Apply "or" condition between Type and Segment filter.
    - If the user asks for rate drops, increases, or comparisons (e.g., “realized rate dropped more than $3”), generate a CTE that computes the KPI for both the current and previous period (typically month or quarter), and then compare them in the final SELECT. Always compute both values explicitly — do not reference non-existent columns like PreviousRealizedRate.
    - Apply ONLY the exact ADD FILTERS from the disambiguated plan for specific values. If no ADD FILTERS, do NOT add any for mentioned values. For thresholds like '< 30', use in HAVING, not as string filters (e.g., no LIKE '%30%').
    - For aliases in SQL, replace special characters like '&' with '_and_' (e.g., 'C&B' becomes 'C_and_B') to avoid syntax errors. Do not use unquoted aliases with special characters.
    MANDATORY RULES:
    1. **No Hallucinated Filters**: Do not add `OR "Type" = 'Cost'` or any filter not explicitly mentioned in the architect's plan.
    2. **KPI Formulas**: When the plan mentions 'Cost', use the SUM of "Amount in USD" where "Group1" matches the specific categories (Direct Expense, etc.).
    3. **Case Sensitivity**: Segment values like 'Transportation' should match the database. Use: WHERE "Segment" = 'Transportation' (ensure correct casing).
    4. **Filtering**: Always use AND to combine Segment filters and Group1 filters. 
       Example: WHERE "Segment" = 'Transportation' AND "Group1" IN ('Direct Expense', 'Indirect Expense')
    5. **Aggregation**: Use SUM("Amount in USD").
    6. **Factor/Breakdown Logic**: 
    - When the plan asks for "factors" or "breakdown":
    - SELECT "Group Description" (or the relevant descriptive column) and the SUM("Amount in USD").
    - ALWAYS add `ORDER BY SUM("Amount in USD") DESC` so the biggest factors appear first.
    - This provides the necessary data for the Insight Agent to explain the 'Why'.
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

def disambiguator_agent(state: AgentState):
    if state.get("is_greeting") or state.get("is_explanation"):
        return {"disambiguated_plan": state.get("architect_plan", ""), "needs_clarification": False}

    potential_values = re.findall(r'\b[a-zA-Z0-9-]{2,}\b', state['question'])
    if not potential_values:
        return {"disambiguated_plan": state["architect_plan"], "needs_clarification": False}
    
    filters = []
    is_account_query = any(word in state['question'].lower() for word in ['account', 'customer', 'finalcustomername'])
    
    for val in set(pv.upper() for pv in potential_values):
        if val in value_to_cols and value_to_cols[val]:
            candidates = value_to_cols[val]
            unique_cols = list(set([c[1] for c in candidates]))

            # --- NEW: CLARIFICATION LOGIC ---
            # If a value matches multiple columns and the user hasn't specified context (is_account_query)
            # and it's not a greeting/explanation, ask for help.
            if len(unique_cols) > 1 and not is_account_query:
                options = [f"Search for '{val}' in {col}" for col in unique_cols]
                return {
                    "needs_clarification": True,
                    "clarification_options": options,
                    "insight": f"I found '{val}' as both a {', '.join(unique_cols[:-1])} and {unique_cols[-1]}. Which one should I use?"
                }
            
            # --- YOUR ORIGINAL PREFERENCE LOGIC ---
            preferred_candidates = [c for c in candidates if c[1] == 'FinalCustomerName'] if is_account_query else candidates
            best = preferred_candidates[0] if preferred_candidates else candidates[0]
            table_alias = 'u' if 'utt' in best[0] else 'p'
            filters.append(f'{table_alias}.\"{best[1]}\" = \'{val}\'')
            print(f"Matched '{val}' → {best[0]}.\"{best[1]}\" ({best[2]} rows)")
            
        else:
            # --- YOUR ORIGINAL FUZZY LOGIC ---
            if len(val) >= 5:
                fuzzy = []
                for v in value_to_cols:
                    if len(get_close_matches(val, [v], n=1, cutoff=0.8)):
                        fuzzy.append(value_to_cols[v][0])
                if fuzzy:
                    best = max(fuzzy, key=lambda x: x[2])
                    filters.append(f'{ "u" if "utt" in best[0] else "p" }.\"{best[1]}\" LIKE \'%{val}%\'')
            else:
                print(f"No exact match for short value '{val}'; skipping fuzzy.")

    if filters:
        filter_str = " AND " + " AND ".join(filters)
        updated_plan = state["architect_plan"] + f"\n\nADD FILTERS: {filter_str}"
    else:
        updated_plan = state["architect_plan"] + "\n\nNo matching columns for filters."
    
    return {
        "disambiguated_plan": updated_plan, 
        "needs_clarification": False,
        "clarification_options": []
    }

# ... (visualizer_agent, insights_agent, graph, UI remain unchanged)
def execute_query_node(state: AgentState):
    if state.get("is_greeting") or state.get("is_explanation") or state.get("query") == "SKIP":
        return {"result": None} # Use None instead of empty DF
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(state["query"], conn)
        conn.close()
        # Convert to dict so MemorySaver can serialize it
        return {"result": df.to_dict(orient='records'), "error": None} 
    except Exception as e:
        return {"error": str(e), "result": None}
 
def visualizer_agent(state: AgentState):
    # 1. Convert List to DataFrame (Memory Compatibility)
    res = state.get("result")
    if isinstance(res, list):
        df_for_viz = pd.DataFrame(res)
    elif isinstance(res, pd.DataFrame):
        df_for_viz = res
    else:
        df_for_viz = pd.DataFrame()

    # 2. Safety Check - Use the converted DataFrame
    if state.get("is_greeting") or state.get("is_explanation") or state.get("error") or df_for_viz.empty: 
        return {"chart_code": ""}
    
    # 3. Use 'df_for_viz' for ALL logic
    cols = list(df_for_viz.columns) # Use the converted DF
    num_rows = len(df_for_viz)      # Use the converted DF
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
   
    # We enforce cols[0] as X (Category) and cols[-1] as Y (Numeric)
    prompt = f"""
Generate Streamlit code to plot this data correctly using plotly.express as px.
The DataFrame is provided as 'df'.
Columns: {cols} (Note: {cols[0]} is the categorical dimension, {cols[-1]} is the numeric KPI).
Number of rows: {num_rows}

Steps to follow:
1. Detect chart type:
   - If columns include 'Month' or date-like, use px.line(df, x='{cols[0]}', y='{cols[-1]}') for time series.
   - For all other categorical queries (like FinalCustomerName), ALWAYS use a vertical bar chart:
     fig = px.bar(df, x='{cols[0]}', y='{cols[-1]}', orientation='v')
2. Sort the df by the numeric column: df = df.sort_values('{cols[-1]}', ascending=False)
3. If the KPI name contains 'cm%' or 'percent', format y-axis as percentage.
4. Update layout:
   - fig.update_layout(
       xaxis_title='{cols[0].replace("_", " ").title()}',  
       yaxis_title='{cols[-1].replace("_", " ").title()}',  
       yaxis_tickformat='.2f%' if 'cm%' in '{cols[-1]}'.lower() or 'percent' in '{cols[-1]}'.lower() else '.2f',
       xaxis_tickangle=-45
   )
5. If time series, sort df by date: df = df.sort_values('{cols[0]}')
6. Add hover_data=['{cols[0]}', '{cols[-1]}'].
7. Finally, call st.plotly_chart(fig, use_container_width=True)
8. Handle empty data: if df.empty, st.write('No data to plot.')
RAW CODE ONLY. No markdown.
    """
    response = llm.invoke(prompt)
    return {"chart_code": response.content.strip().replace("```python", "").replace("```", "")}
 
def insights_agent(state: AgentState):
    # 1. Handle Greetings or Explanations
    if state.get("is_greeting") or state.get("is_explanation"): 
        # Ensure we don't return None; use a fallback string if insight is missing
        return {"insight": state.get("insight", "How else can I help you today?")}

    # 2. Handle the 'List vs DataFrame' Memory Issue (Correctly implemented)
    res = state.get("result")
    if isinstance(res, list):
        df = pd.DataFrame(res)
    elif isinstance(res, pd.DataFrame):
        df = res
    else:
        df = pd.DataFrame()

    # 3. If no data was found
    if df.empty:
        return {"insight": "I couldn't find any data matching your specific filters."}

    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    
    # --- FIXED: Use the 'df' variable here, NOT 'state['result']' ---
    prompt = f"""
    Analyze the following data result for the user's question: "{state['question']}"
    
    DATA:
    {df.to_string(index=False)}
    
    INSTRUCTIONS:
    - If the data is a breakdown (multiple rows), identify the top 2-3 highest factors and explain their impact.
    - If the values are negative, mention if they represent credits or potential data anomalies.
    - Keep it professional and focused on business impact.
    - Limit to 3 concise sentences.
    """
    
    try:
        response = llm.invoke(prompt)
        return {"insight": response.content.strip()}
    except Exception as e:
        return {"insight": "Data retrieved successfully, but insight generation failed."}
# --- 5. GRAPH ---
builder = StateGraph(AgentState)

# 1. Add all nodes
builder.add_node("architect", architect_agent)
builder.add_node("disambiguator", disambiguator_agent) 
builder.add_node("sql_analyst", sql_analyst_agent)
builder.add_node("executor", execute_query_node)
builder.add_node("visualizer", visualizer_agent)
builder.add_node("insights", insights_agent)

# 2. Define the flow (Edges)
builder.add_edge(START, "architect")
builder.add_edge("architect", "disambiguator")
builder.add_edge("disambiguator", "sql_analyst")
builder.add_edge("sql_analyst", "executor")
builder.add_edge("executor", "visualizer")
builder.add_edge("visualizer", "insights")
builder.add_edge("insights", END)

# 3. Initialize memory
memory = MemorySaver()

# 4. Compile (Do NOT re-initialize builder before this!)
graph = builder.compile(checkpointer=memory)


# --- 1. GLOBAL CONFIG & PERSISTENCE ---
# --- 1. GLOBAL CONFIG & PERSISTENCE ---


st.set_page_config(page_title="Financial AI Agent", layout="wide", page_icon="🏛️")

# Ensure these functions (load_history, save_history, delete_chat_session, clear_all_history_file) 
# are imported from your backend script or defined above this line.

if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_history()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

# --- 2. PROFESSIONAL UI STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    .stApp { background-color: #ffffff; font-family: 'Roboto', sans-serif; }

    [data-testid="stSidebar"] {
        background-color: #f0f4f9 !important;
        border-right: 1px solid #e3e3e3;
        width: 300px !important;
    }

    .sidebar-header {
        font-size: 0.75rem; 
        font-weight: 600; 
        color: #1a73e8;
        text-transform: uppercase; 
        letter-spacing: 0.08rem;
        margin-top: 2.5rem !important;
        margin-bottom: 0.8rem !important;
        padding-left: 0.5rem;
    }

    .first-header { margin-top: 1rem !important; }

    .stButton > button {
        border: none !important; 
        background-color: transparent !important; 
        color: #444746 !important;
        transition: all 0.2s ease-in-out;
        border-radius: 8px !important; 
        text-align: left !important;
        font-size: 0.9rem; 
        padding: 0.6rem 1rem !important;
        width: 100%; 
        display: flex; 
        justify-content: flex-start;
    }

    .stButton > button[kind="primary"] {
        background-color: #d3e3fd !important; 
        color: #041e49 !important;
    }

    .stButton > button:hover {
        background-color: #e8f0fe !important; 
        color: #1a73e8 !important;
        transform: translateX(2px);
    }

    div[data-testid="stPopover"] {
        background-color: transparent !important;
    }

    div[data-testid="stPopover"] button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #5f6368 !important;
        padding: 0 !important;
        width: 24px !important;
        height: 24px !important;
    }
    
    .stExpander details summary svg {
        display: none !important;
    }
    
    .stExpander {
        border: none !important;
        background-color: transparent !important;
    }

    [data-testid="stVerticalBlock"] { gap: 0.1rem !important; }

    .main-title {
        font-weight: 500; font-size: 1.6rem; color: #1f1f1f;
        margin: 1rem 0 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR: NAVIGATION ---
with st.sidebar:
    st.markdown("<div style='font-size: 1.3rem; font-weight: 500; color: #1a73e8; padding: 1.5rem 0.5rem;'>🏛️ Financial IQ</div>", unsafe_allow_html=True)
    
    # --- ACTIONS ---
    st.markdown("<p class='sidebar-header first-header'>Actions</p>", unsafe_allow_html=True)
    if st.button("➕ New Chat" \
    "", use_container_width=True, type="primary"):
        if st.session_state.messages:
            st.session_state.all_chats[st.session_state.thread_id] = {
                "messages": st.session_state.messages,
                "updated_at": datetime.now()
            }
            save_history()
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    # --- HISTORY ---
    st.markdown("<p class='sidebar-header'>History</p>", unsafe_allow_html=True)
    with st.expander("📂 Recent Chats", expanded=False):
        if not st.session_state.all_chats:
            st.info("No history yet.")
        else:
            sorted_chats = sorted(
                st.session_state.all_chats.items(), 
                key=lambda x: x[1].get("updated_at", datetime.now()), 
                reverse=True
            )
            
            for tid, data in sorted_chats:
                msgs = data["messages"]
                first_q = next((m["content"] for m in msgs if m["role"] == "user"), "New Analysis")
                title = str(first_q)[:18] + "..." if len(str(first_q)) > 18 else str(first_q)
                
                # Logic for "Time Ago" label
                updated_at = data.get("updated_at", datetime.now())
                delta = datetime.now() - updated_at
                if delta.days >= 1:
                    time_str = f"{delta.days}d"
                elif delta.seconds // 3600 >= 1:
                    time_str = f"{delta.seconds // 3600}h"
                else:
                    time_str = f"{delta.seconds // 60}m"

                c1, c2 = st.columns([0.85, 0.15])
                with c1:
                    is_active = tid == st.session_state.thread_id
                    if st.button(f" {title} • {time_str}", key=f"btn_{tid}", use_container_width=True, type="primary" if is_active else "secondary"):
                        st.session_state.thread_id = tid
                        st.session_state.messages = msgs
                        st.rerun()
                with c2:
                    with st.popover("⋮"):
                        # USE YOUR ORIGINAL BACKEND FUNCTION HERE
                        if st.button("🗑️ Delete", key=f"del_{tid}"):
                            delete_chat_session(tid) # This calls your actual file deletion logic
                            if tid == st.session_state.thread_id:
                                st.session_state.thread_id = str(uuid.uuid4())
                                st.session_state.messages = []
                            st.rerun()

    # --- SYSTEM ---
    st.markdown("<p class='sidebar-header'>System</p>", unsafe_allow_html=True)
    with st.expander("⚙️ Settings", expanded=False):
        if st.button("🗑️ Clear All History", use_container_width=True):
            clear_all_history_file() # This calls your actual file clearing logic
            st.session_state.all_chats = {}
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
# --- 4. MAIN CONTENT ---
st.markdown("<h1 class='main-title'>Financial Intelligence Engine</h1>", unsafe_allow_html=True)

schema_info = db.get_table_info()

# Display current conversation
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🏛️"
    with st.chat_message(msg["role"], avatar=avatar):
        if isinstance(msg["content"], pd.DataFrame): 
            st.dataframe(msg["content"], use_container_width=True, hide_index=True)
        elif isinstance(msg["content"], dict) and msg["content"].get("is_chart"):
            try:
                chart_df = msg["content"]["df"]
                chart_code = msg["content"]["code"]
                exec(chart_code, globals(), {'st': st, 'df': chart_df, 'px': px})
            except Exception:
                st.info("Visual rendering unavailable.")
        else: 
            st.markdown(msg["content"])

# --- 5. CHAT INPUT & AGENT LOGIC ---
if prompt := st.chat_input("Ask about your finances..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"): 
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Prepare memory for LLM
            chat_history = []
            for m in st.session_state.messages[-6:]: 
                role = "User" if m["role"] == "user" else "Assistant"
                content = "[Data Table]" if isinstance(m["content"], pd.DataFrame) else m["content"]
                chat_history.append(f"{role}: {content}")

            # Graph Call
            output = graph.invoke(
                {"question": prompt, "schema": schema_info, "history": chat_history}, 
                config=config
            )
            
            # Handle Clarification
            if output.get("needs_clarification"):
                st.markdown(f"🤔 **{output['insight']}**")
                cols = st.columns(len(output["clarification_options"]))
                for idx, option in enumerate(output["clarification_options"]):
                    if cols[idx].button(option, key=f"clarify_{idx}"):
                        st.session_state.messages.append({"role": "user", "content": option})
                        st.rerun()
                st.stop()

            # Result Processing
            elif output.get("error"):
                st.error(f"Error: {output['error']}")
            
            elif output.get("is_greeting") or output.get("is_explanation"):
                st.markdown(output["insight"])
                st.session_state.messages.append({"role": "assistant", "content": output["insight"]})

            else:
                data_from_state = output.get("result")
                raw_df = pd.DataFrame(data_from_state) if isinstance(data_from_state, list) else data_from_state
                
                if raw_df is not None and not raw_df.empty:
                    display_df = raw_df.copy()
                    # Formatting numbers
                    for col in display_df.select_dtypes(include=['number']).columns:
                        if 'percent' in col.lower() or 'cm%' in col.lower():
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
                        else:
                            display_df[col] = display_df[col].round(2)

                    st.markdown("#### 📊 Data Insights")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    st.session_state.messages.append({"role": "assistant", "content": display_df})
                    
                    if output.get("chart_code"):
                        st.markdown("#### 📈 Visualization")
                        try:
                            chart_df = raw_df.copy()
                            for col in chart_df.columns:
                                if col != chart_df.columns[0]:
                                    chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
                            
                            local_vars = {'st': st, 'df': chart_df, 'px': px}
                            exec(output["chart_code"], globals(), local_vars)
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": {"is_chart": True, "code": output["chart_code"], "df": chart_df}
                            })
                        except Exception as chart_err:
                            st.info("Chart skipped.")

                if output.get("insight"):
                    st.success(output['insight'])
                    st.session_state.messages.append({"role": "assistant", "content": output["insight"]})

                if output.get("query") and output["query"] != "SKIP":
                    with st.expander("Technical SQL Log"):
                        st.code(output["query"], language="sql")

            # Final Save
            st.session_state.all_chats[st.session_state.thread_id] = {
                "messages": st.session_state.messages,
                "updated_at": datetime.now()
            }
            save_history()
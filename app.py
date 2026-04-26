import os
import sys
from pathlib import Path
import streamlit as st
import json
import requests

@st.cache_data(ttl=300, show_spinner="Fetching closed PRs from GitHub...")
def fetch_closed_prs(owner: str, repo: str, token: str):
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=closed&per_page=15"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        prs_list = response.json()
        return [{"number": pr["number"], "title": pr["title"]} for pr in prs_list]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch PRs: {e}")
        return []

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try to import workflow components
try:
    from src.Workflow import Workflow
    from templates.state import prState
except ImportError as e:
    st.error(f"Failed to import Proxima modules. Error: {e}")

st.set_page_config(page_title="Sphinx PR Review Simulator", layout="wide")
# --- SIDEBAR: Configuration ---
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    st.subheader("1. Data Generation Model")
    st.markdown("Generates the Review Checklist")
    data_provider = st.selectbox("Data Provider", ["Google Gemini", "Groq"], index=0, key="data_provider")
    data_model = st.text_input("Model Name", value="gemini-3.1-flash-lite-preview", key="data_model")
    data_api_key = st.text_input("API Key", type="password", key="data_api_key")
    
    st.divider()
    
    st.subheader("2. Model Under Evaluation")
    st.markdown("Generates the PR Review to be Scored")
    eval_provider = st.selectbox("Evaluation Provider", ["Groq", "Google Gemini"], index=0, key="eval_provider")
    eval_model = st.text_input("Model Name", value="meta-llama/llama-4-scout-17b-16e-instruct", key="eval_model")
    eval_api_key = st.text_input("Evaluation API Key", type="password", key="eval_api_key")

# --- MAIN CONTENT / EDUCATIONAL HEADER ---
st.title("Proxima: AI Model Benchmarking Simulator")
st.markdown("""
Welcome to **Proxima**, the enterprise framework for evaluating how well different LLMs perform code reviews. 

1. **The Ground Truth** (*Data Generation*): We reverse-engineer a real, merged Pull Request to find the *actual* bugs the human fixed.
2. **The Test** (*Model Under Evaluation*): We ask your selected AI Model to review the broken code.
3. **The Score**: We mathematically calculate what percentage of the actual, real-world bugs the AI successfully caught natively.
""")
st.divider()

st.subheader("Repository Details")

col1, col2 = st.columns(2)
with col1:
    repo_owner = st.text_input("Owner (e.g., langchain-ai)", value="dbader")
with col2:
    repo_name = st.text_input("Repository (e.g., schedule)", value="schedule")

pr_number = 0

if st.button("Fetch Closed PRs", help="Pull the 15 most recent closed PRs from GitHub"):
    if repo_owner and repo_name:
        prs = fetch_closed_prs(repo_owner, repo_name, os.getenv("GITHUB_TOKEN"))
        st.session_state.fetched_prs = prs
    else:
        st.warning("Please enter an Owner and Repository name.")

if "fetched_prs" in st.session_state and st.session_state.fetched_prs:
    pr_options = {f"PR #{item['number']}: {item['title']}": item['number'] for item in st.session_state.fetched_prs}
    selected_pr_label = st.selectbox("Choose a Pull Request to Benchmark", list(pr_options.keys()))
    pr_number = pr_options.get(selected_pr_label, 0)

# State management for results
if "workflow_completed" not in st.session_state:
    st.session_state.workflow_completed = False
if "final_state" not in st.session_state:
    st.session_state.final_state = None

st.divider()

# Custom CSS for nicer fonts
st.markdown("""
    <style>
    .checklist-item {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        line-height: 1.6;
        padding: 12px;
        margin-bottom: 12px;
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        border-radius: 4px;
        color: #1E1E1E;
    }
    .metric-value {
        font-size: 2.5rem !important;
        font-weight: 700;
        color: #2e66ff;
    }
    </style>
""", unsafe_allow_html=True)

# --- Run Workflows ---

if st.button("Run Workflow", type="primary", width="stretch"):
    # Reset state at the start of a run so old outputs don't show on crash
    st.session_state.workflow_completed = False
    st.session_state.final_state = None

    if not os.getenv("GITHUB_TOKEN"):
        st.warning("Please ensure GITHUB_TOKEN is set in your .env file or environment.")
    else:
        from utils.models import configure_google_model, configure_groq_model

        # Configure models based on the sidebar configs
        if data_provider == "Google Gemini":
            configure_google_model(api_key=data_api_key, model_name=data_model)
        elif data_provider == "Groq":
            configure_groq_model(api_key=data_api_key, model_name=data_model)
            
        if eval_provider == "Groq":
            configure_groq_model(api_key=eval_api_key, model_name=eval_model) 
        elif eval_provider == "Google Gemini":
            configure_google_model(api_key=eval_api_key, model_name=eval_model)
            
        with st.status("Executing Sequential Workflows...", expanded=True) as status:
            st.write("Initializing Workflow...")
            try:
                pr_num_val = int(pr_number) if pr_number > 0 else None
                
                # Pre-flight check: Reject PRs with > 5 changes immediately to save token costs
                if pr_num_val:
                    st.write("Validating Pull Request dimensions...")
                    detail_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_num_val}"
                    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"}
                    detail_resp = requests.get(detail_url, headers=headers)
                    if detail_resp.status_code == 200:
                        detail = detail_resp.json()
                        changed_files = detail.get("changed_files", 0)
                        if changed_files > 5:
                            raise ValueError(f"PR #{pr_num_val} consists of {changed_files} modified files, which strictly exceeds the 5-file safety limit to prevent hallucination and token exhaustion. Please select a smaller scale PR.")

                initial_state = prState(owner=repo_owner, repo=repo_name, pr_number=pr_num_val)
                
                # Execute full sequential run (Data Gen -> Eval)
                workflow = Workflow(
                    data_provider=data_provider,
                    data_model=data_model,
                    eval_provider=eval_provider,
                    eval_model=eval_model
                )
                st.write("Executing LangGraph nodes... (This may take a minute)")
                final_state = workflow.graph.invoke(initial_state)
                
                st.session_state.final_state = final_state
                st.session_state.workflow_completed = True
                status.update(label="Workflow Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Task failed", state="error")
                st.error(f"Error: {str(e)}")

# --- Displays Outputs ---
if st.session_state.workflow_completed and st.session_state.final_state:
    st.success("Workflow finished executing.")
    
    st.header("Results")
    
    # Extract keys safely from dict or State model
    final_dict = st.session_state.final_state
    if hasattr(final_dict, "model_dump"):
        final_dict = final_dict.model_dump()
        
    review_data = final_dict.get("llm_generated_pr_review")
    checklist_data = final_dict.get("checklist")
    score_data = final_dict.get("score")
    
    tab1, tab2, tab3 = st.tabs(["LLM Generated Review", "Review Checklist", "Evaluation Score"])
    
    with tab1:
        if review_data:
            r_dict = review_data if isinstance(review_data, dict) else review_data.model_dump()
            st.markdown(f"### **Summary:**\n{r_dict.get('summary', '')}")
            st.divider()
            
            for index, c in enumerate(r_dict.get('comments', [])):
                sev = c.get('severity', 'LOW').upper()
                
                with st.expander(f"[{sev}] {c.get('file_path', '')} (Line {c.get('line_reference', 'N/A')})"):
                    st.markdown(f"**Category:** `{c.get('category', 'General')}`")
                    st.write(c.get('comment', ''))
                    
                    if c.get('suggestion'):
                        # Using info banner instead of python code block for plain english
                        st.info(f"**Suggestion:** {c.get('suggestion')}")
        else:
            st.info("No LLM PR Review was generated.")
            
    with tab2:
        if checklist_data:
            c_dict = checklist_data if isinstance(checklist_data, dict) else checklist_data.model_dump()
            st.markdown(f"### **Overview**\n* {c_dict.get('summary', '')}")
            st.divider()
            
            items = c_dict.get('items', [])
            if items:
                for idx, item in enumerate(items):
                    sev = item.get('severity', 'low').upper()
                    # Render dynamically grouped, beautiful checklist cards
                    st.markdown(f"""
                        <div class="checklist-item">
                            <b>Item {idx+1} ({sev})</b> — <code>{item.get('file_path', '')}</code><br/>
                            <b>Target:</b> {item.get('target', '')}<br/>
                            <b>Verify:</b> {item.get('verification_point', '')}<br/>
                            <b>Expected:</b> {item.get('expected_outcome', '')}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No items found in checklist.")
        else:
            st.info("No Review Checklist was generated.")
            
    with tab3:
        if score_data:
            s_dict = score_data if isinstance(score_data, dict) else score_data.model_dump()
            
            # Extract variables safely
            matched = int(s_dict.get('matched_items', 0))
            total = int(s_dict.get('total_items', 0))
            coverage = float(s_dict.get('coverage_score', 0)) * 100
            
            st.markdown("### **Final Evaluation Profile**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-value'>{matched}</div><div>Matched Items</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-value'>{total}</div><div>Total Items Required</div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-value'>{coverage:.1f}%</div><div>Coverage Score</div>", unsafe_allow_html=True)
            
            st.divider()
            st.write(f"**Methodology:** Evaluated utilizing `{s_dict.get('evaluation_mode', 'Unknown')}` mode.")
        else:
            st.info("No Evaluation Score was generated.")

import os
import sys
from dotenv import load_dotenv

# Add src to python path so it can import Agents
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.Workflow import Workflow
from templates.state import prState

def main():
    load_dotenv()
    
    github_token = os.getenv("GITHUB_TOKEN")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    owner = os.getenv("TEST_REPO_OWNER", "langchain-ai")
    repo = os.getenv("TEST_REPO_NAME", "langchain")
    pr_number = int(os.getenv("TEST_PR_NUMBER", "36676"))
    
    if not github_token:
        print("Missing GITHUB_TOKEN in env.")
        return
        
    workflow = Workflow(
        github_token=github_token,
        api_key=api_key,
    )
    
    print(f"Initializing test run on {owner}/{repo} PR #{pr_number}...")
    initial_state = prState(owner=owner, repo=repo, pr_number=pr_number)
    
    final_state = workflow.graph.invoke(initial_state)
    print("\n==================\nWorkflow Finished!\n==================")
    print("Check data folder for outputs.")

if __name__ == "__main__":
    main()

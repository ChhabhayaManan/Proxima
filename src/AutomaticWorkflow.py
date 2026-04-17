import csv
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Wait, I need github to fetch closed PRs.
from github import Auth, Github
from github.GithubException import GithubException

from src.DataGenerationWorkflow import DataGenerationWorkflow
from src.ModelEvaluationWorkflow import ModelEvaluationWorkflow
from templates.state import prState
from utils.models import configure_ollama_model

load_dotenv()


def main() -> None:
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable is not set. Please set it and run again.")
        sys.exit(1)

    print("Checking Ollama availability...")
    try:
        import requests
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(base_url, timeout=5)
        response.raise_for_status()
    except Exception as e:
        print(f"Error: Ollama server at {base_url} is unreachable. Exception: {e}")
        print("Please ensure the Ollama server is running offline before starting.")
        sys.exit(1)

    owner_repo = input("Enter repo as owner/name (e.g., Mintplex-Labs/anything-llm): ").strip()
    if "/" not in owner_repo:
        print("Error: Repo must be formatted as owner/name.")
        sys.exit(1)
    
    owner, repo_name = owner_repo.split("/", 1)
    
    data_gen_model = input("Enter Ollama model for data generation/scoring (default: llama3.1): ").strip() or "llama3.1"
    review_model = input("Enter Ollama model for PR review generation (default: llama3.1): ").strip() or "llama3.1"

    configure_ollama_model(model_name=data_gen_model)

    print("Initializing GitHub connection...")
    auth = Auth.Token(github_token)
    g = Github(auth=auth)

    try:
        repo = g.get_repo(f"{owner}/{repo_name}")
    except GithubException as e:
        print(f"Error accessing repo {owner}/{repo_name}: {e}")
        sys.exit(1)

    target_process_input = input("How many eligible PRs do you want to successfully evaluate? (default: 10): ").strip()
    target_process_count = int(target_process_input) if target_process_input.isdigit() else 10

    print(f"Initializing workflows...")
    data_workflow = DataGenerationWorkflow(github_token=github_token, provider="ollama")
    eval_workflow = ModelEvaluationWorkflow(provider="ollama")

    csv_dir = os.path.join(REPO_ROOT, "data", f"{owner}_{repo_name}")
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = os.path.join(csv_dir, "automatic_results.csv")

    file_exists = os.path.isfile(csv_file)

    csv_columns = [
        "processed_at", "pr_number", "pr_title", "review_model", 
        "matched_items", "total_items", "coverage_score", "evaluation_mode", 
        "status", "checklist_items_text", "review_summary"
    ]

    print(f"Fetching PR stream from {owner}/{repo_name}...")
    try:
        pulls = repo.get_pulls(state="closed", sort="updated", direction="desc")
    except Exception as e:
        print(f"Error fetching pull requests: {e}")
        sys.exit(1)

    processed_count = 0
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        for pr in pulls:
            if processed_count >= target_process_count:
                print(f"\nReached target of {target_process_count} processed PRs.")
                break

            try:
                changed_files = pr.changed_files
            except Exception as e:
                print(f"Error accessing PR #{pr.number} stats: {e}")
                continue

            if changed_files > 5 or changed_files == 0:
                print(f"Skipping PR #{pr.number}: {changed_files} changed files.")
                continue

            print(f"\n=============================================")
            print(f"Processing PR #{pr.number}: {pr.title} [{processed_count+1}/{target_process_count}]")
            print(f"=============================================")
            
            row = {
                "processed_at": datetime.now().isoformat(),
                "pr_number": pr.number,
                "pr_title": pr.title,
                "review_model": review_model,
                "status": "success",
            }

            try:
                # 1. New State for each PR
                state = prState(owner=owner, repo=repo_name, pr_number=pr.number)
                
                # 2. Data Generation
                configure_ollama_model(model_name=data_gen_model)
                out_state = data_workflow.graph.invoke(state)

                eval_state = prState.model_validate(out_state)

                # 3. Model Evaluation
                configure_ollama_model(model_name=review_model)
                final_state = eval_workflow.graph.invoke(eval_state)
                
                # Populate results
                if final_state.get("checklist"):
                    checklist = final_state["checklist"]
                    items_text = []
                    if hasattr(checklist, "items") and checklist.items:
                        for i, item in enumerate(checklist.items):
                            items_text.append(f"{i+1}. [{item.severity}] {item.file_path}: {item.target} - {item.verification_point} -> {item.expected_outcome}")
                    row["checklist_items_text"] = "\n".join(items_text)

                if final_state.get("llm_generated_pr_review"):
                    review = final_state["llm_generated_pr_review"]
                    row["review_summary"] = review.summary

                if final_state.get("score"):
                    score = final_state["score"]
                    row["matched_items"] = score.matched_items
                    row["total_items"] = score.total_items
                    row["coverage_score"] = score.coverage_score
                    row["evaluation_mode"] = score.evaluation_mode

            except Exception as e:
                print(f"Error processing PR #{pr.number}: {e}")
                traceback.print_exc()
                row["status"] = f"error - {str(e)}"

            writer.writerow(row)
            f.flush()
            processed_count += 1

    print("\nWorkflow completed.")

if __name__ == "__main__":
    main()

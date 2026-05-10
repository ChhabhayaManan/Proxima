import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any
from datetime import datetime

from dotenv import load_dotenv
from github import Auth, Github


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Workflow import Workflow
from templates.state import prState
from utils.models import (
    DEFAULT_GROQ_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    configure_google_model,
    configure_groq_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PR review pipeline in batch and produce a summary CSV."
    )
    parser.add_argument("--owner", default="pathwaycom")
    parser.add_argument("--repo", default="llm-app")
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Maximum number of merged PRs to process. Use 0 to process all matching PRs.",
    )
    parser.add_argument(
        "--max-changed-files",
        type=int,
        default=None,
        help="Optional filter to skip PRs with more than this many changed files.",
    )
    parser.add_argument(
        "--pr-numbers",
        nargs="*",
        type=int,
        help="Optional explicit PR numbers to process instead of auto-selecting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip PRs that already have score.json.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path for the output CSV summary.",
    )
    parser.add_argument(
        "--google-model",
        default=DEFAULT_MODEL_NAME,
        help=f"Google model to use for Gemini stages. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--groq-model",
        default=DEFAULT_GROQ_MODEL_NAME,
        help=f"Groq model to use for evaluation review generation. Default: {DEFAULT_GROQ_MODEL_NAME}",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def initialize_runtime(args: argparse.Namespace) -> None:
    load_dotenv()
    github_token = require_env("GITHUB_TOKEN")
    google_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not google_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in the environment.")
    groq_key = require_env("GROQ_API_KEY")

    os.environ["GITHUB_TOKEN"] = github_token
    configure_google_model(api_key=google_key, model_name=args.google_model)
    configure_groq_model(api_key=groq_key, model_name=args.groq_model)


def get_github_client() -> Github:
    token = require_env("GITHUB_TOKEN")
    return Github(auth=Auth.Token(token))


def select_prs(
    gh: Github,
    owner: str,
    repo_name: str,
    limit: int,
    max_changed_files: int | None,
    explicit_pr_numbers: list[int] | None,
) -> list[Any]:
    repo = gh.get_repo(f"{owner}/{repo_name}")

    if explicit_pr_numbers:
        return [repo.get_pull(number) for number in explicit_pr_numbers]

    selected = []
    for pr in repo.get_pulls(state="closed", sort="created", direction="asc"):
        if pr.merged_at is None:
            continue
        if max_changed_files is not None and pr.changed_files > max_changed_files:
            continue

        selected.append(pr)
        if limit and len(selected) >= limit:
            break

    return selected


def output_folder(owner: str, repo: str, pr_number: int) -> Path:
    return REPO_ROOT / "data" / f"{owner}_{repo}_pr{pr_number}"


def load_json_if_exists(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_summary_row(owner: str, repo: str, pr: Any, status: str, error: str = "") -> dict[str, Any]:
    folder = output_folder(owner, repo, pr.number)
    score = load_json_if_exists(folder / "score.json") or {}
    checklist = load_json_if_exists(folder / "review_checklist.json") or {}
    llm_review = load_json_if_exists(folder / "llm_generated_pr_review.json") or {}

    checklist_items = checklist.get("items", []) if isinstance(checklist, dict) else []
    llm_comments = llm_review.get("comments", []) if isinstance(llm_review, dict) else []

    return {
        "pr_number": pr.number,
        "title": pr.title or "",
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else "",
        "changed_files": pr.changed_files,
        "additions": pr.additions,
        "deletions": pr.deletions,
        "status": status,
        "error": error,
        "checklist_items": len(checklist_items),
        "llm_review_comments": len(llm_comments),
        "matched_items": score.get("matched_items", ""),
        "total_items": score.get("total_items", ""),
        "coverage_score": score.get("coverage_score", ""),
        "evaluation_mode": score.get("evaluation_mode", ""),
        "A_avg": "",
        "B_avg": "",
        "C_avg": "",
        "overall_avg": "",
        "notes": "",
    }


def write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> Path:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pr_number",
        "title",
        "merged_at",
        "changed_files",
        "additions",
        "deletions",
        "status",
        "error",
        "checklist_items",
        "llm_review_comments",
        "matched_items",
        "total_items",
        "coverage_score",
        "evaluation_mode",
        "A_avg",
        "B_avg",
        "C_avg",
        "overall_avg",
        "notes",
    ]
    try:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return summary_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = summary_path.with_name(
            f"{summary_path.stem}_autosave_{timestamp}{summary_path.suffix}"
        )
        with fallback_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"Summary file was locked: {summary_path}. Wrote autosave copy instead: {fallback_path}"
        )
        return fallback_path


def main() -> None:
    args = parse_args()
    initialize_runtime(args)

    gh = get_github_client()
    workflow = Workflow()
    prs = select_prs(
        gh=gh,
        owner=args.owner,
        repo_name=args.repo,
        limit=args.limit,
        max_changed_files=args.max_changed_files,
        explicit_pr_numbers=args.pr_numbers,
    )

    if not prs:
        print("No PRs matched the current selection criteria.")
        return

    summary_path = Path(args.summary_path) if args.summary_path else (
        REPO_ROOT / "data" / f"{args.owner}_{args.repo}_batch_summary.csv"
    )

    rows: list[dict[str, Any]] = []
    total = len(prs)
    for index, pr in enumerate(prs, start=1):
        folder = output_folder(args.owner, args.repo, pr.number)
        score_path = folder / "score.json"

        print(f"[{index}/{total}] Processing PR #{pr.number}: {pr.title}")
        if args.resume and score_path.exists():
            print("  Skipping because score.json already exists.")
            rows.append(build_summary_row(args.owner, args.repo, pr, status="skipped"))
            continue

        try:
            state = prState(owner=args.owner, repo=args.repo, pr_number=pr.number)
            workflow.graph.invoke(state)
            rows.append(build_summary_row(args.owner, args.repo, pr, status="success"))
        except Exception as exc:
            print(f"  Failed: {exc}")
            rows.append(build_summary_row(args.owner, args.repo, pr, status="failed", error=str(exc)))

        summary_path = write_summary(summary_path, rows)

    print(f"\nBatch run complete. Summary written to: {summary_path}")


if __name__ == "__main__":
    main()

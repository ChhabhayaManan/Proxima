import json
import os
import re

from dotenv import load_dotenv
from github import Auth, Github
from github.GithubException import GithubException, UnknownObjectException

from templates.state import prState


load_dotenv()


class gitPRRetriever:
    def __init__(self, token: str):
        if not token:
            raise ValueError("GitHub token must be provided.")

        auth = Auth.Token(token)
        self.g = Github(auth=auth)

        if not self.g.get_user().login:
            raise ValueError("Invalid GitHub token provided.")

    def run(self, state: prState) -> dict:
        repo = self.g.get_repo(f"{state.owner}/{state.repo}")
        if state.pr_number is not None and state.pr_number > 0:
            selected_pr = repo.get_pull(int(state.pr_number))
        else:
            selected_pr = self.select_pr(repo)

        print(
            f"Selected PR: {selected_pr.number} {selected_pr.title}, changed files: {selected_pr.changed_files}"
        )

        data, pr_details, issue_details = self.collect_pr_artifacts(repo, selected_pr)
        self.store_data(data, state.owner, repo.name, selected_pr.number, pr_details, issue_details)

        state.pr_number = selected_pr.number
        return state.model_dump()

    def select_pr(self, repo):
        candidate_prs = self.list_candidate_prs(repo)
        if not candidate_prs:
            print("No recent closed PRs matched the default filter. You can still enter any closed PR number.")

        while True:
            input_pr_number = input("Enter the PR number that you want to review: ").strip()
            if not input_pr_number.isdigit():
                print("Please enter a numeric PR number.")
                continue

            try:
                pr = repo.get_pull(int(input_pr_number))
            except UnknownObjectException:
                print("That PR number does not exist in this repository. Try again.")
                continue

            if pr.state != "closed":
                print("Please choose a closed PR so the workflow can compare the base and merged changes.")
                continue

            return pr

    def list_candidate_prs(self, repo, max_candidates: int = 10):
        print(f"Repository: {repo.full_name}")
        print("=" * 100)
        print("Recent closed PRs with fewer than 5 changed files:")
        print("PR number : title")

        candidate_prs = []
        for pr in repo.get_pulls(state="closed", sort="created", direction="asc"):
            if pr.changed_files >= 5:
                continue

            candidate_prs.append(pr)
            print(
                f"PR {pr.number}: {pr.title} , additions: {pr.additions} , deletions: {pr.deletions}, "
                f"changed files: {pr.changed_files} , state: {pr.state}"
            )

            if len(candidate_prs) >= max_candidates:
                break

        print("=" * 100)
        return candidate_prs

    def collect_pr_artifacts(self, repo, pr):
        head_ref = pr.head.sha
        base_ref = pr.base.sha
        comparison = repo.compare(base_ref, head_ref)
        patches_by_filename = {
            compared_file.filename: compared_file.patch or ""
            for compared_file in comparison.files
        }

        data = {}
        for file in pr.get_files():
            print(
                f"File: {file.filename}, additions: {file.additions}, deletions: {file.deletions}"
            )
            data[file.filename] = {
                "base_content": self.get_file_content(repo, file.filename, base_ref),
                "head_content": self.get_file_content(repo, file.filename, head_ref),
                "changed_lines": patches_by_filename.get(file.filename, ""),
            }

        issue_comments = []
        for comment in pr.get_issue_comments():
            issue_comments.append(
                {
                    "user": comment.user.login if comment.user else None,
                    "body": comment.body or "",
                    "created_at": comment.created_at.isoformat() if comment.created_at else None,
                    "updated_at": comment.updated_at.isoformat() if comment.updated_at else None,
                }
            )

        review_comments = []
        for comment in pr.get_review_comments():
            review_comments.append(
                {
                    "user": comment.user.login if comment.user else None,
                    "body": comment.body or "",
                    "path": comment.path,
                    "line": comment.line,
                    "created_at": comment.created_at.isoformat() if comment.created_at else None,
                    "updated_at": comment.updated_at.isoformat() if comment.updated_at else None,
                }
            )

        pr_details = {
            "pr_number": pr.number,
            "title": pr.title or "",
            "description": pr.body or "",
            "comments": {
                "issue_comments": issue_comments,
                "review_comments": review_comments,
            },
        }
        issue_details = self.get_related_issue_details(repo, pr, pr_details)

        return data, pr_details, issue_details

    def get_file_content(self, repo, filename: str, ref: str) -> str:
        try:
            content = repo.get_contents(filename, ref=ref)
        except UnknownObjectException:
            return ""
        except GithubException as exc:
            print(f"Unable to fetch {filename} at {ref}: {exc}")
            return ""

        if isinstance(content, list):
            return ""

        try:
            return content.decoded_content.decode("utf-8").strip()
        except UnicodeDecodeError:
            print(f"Skipping non-text content for {filename} at {ref}.")
            return ""

    def get_related_issue_details(self, repo, pr, pr_details):
        text_blobs = [
            pr.title or "",
            pr.body or "",
            pr_details.get("description", ""),
        ]

        issue_comments = pr_details.get("comments", {}).get("issue_comments", [])
        review_comments = pr_details.get("comments", {}).get("review_comments", [])
        for comment in issue_comments:
            text_blobs.append(comment.get("body", ""))
        for comment in review_comments:
            text_blobs.append(comment.get("body", ""))

        linked_numbers = self.extract_issue_numbers(
            "\n".join(text_blobs), repo.owner.login, repo.name
        )

        issue_items = []
        for number in sorted(linked_numbers):
            if number == pr.number:
                continue

            try:
                issue = repo.get_issue(number)

                if issue.pull_request is not None:
                    continue

                issue_items.append(
                    {
                        "number": issue.number,
                        "title": issue.title or "",
                        "body": issue.body or "",
                        "state": issue.state,
                        "user": issue.user.login if issue.user else None,
                        "labels": [label.name for label in issue.labels],
                        "assignees": [assignee.login for assignee in issue.assignees],
                        "comments_count": issue.comments,
                        "created_at": issue.created_at.isoformat() if issue.created_at else None,
                        "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
                        "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
                        "html_url": issue.html_url,
                    }
                )
            except GithubException as exc:
                print(f"Unable to fetch related issue #{number}: {exc}")

        return {
            "related_issue_numbers": sorted([item["number"] for item in issue_items]),
            "issues": issue_items,
        }

    def extract_issue_numbers(self, text, owner, repo_name):
        issue_numbers = set()
        if not text:
            return issue_numbers

        patterns = [
            rf"(?<![A-Za-z0-9_])#(\d+)\b",
            rf"\b{re.escape(owner)}/{re.escape(repo_name)}#(\d+)\b",
            rf"https?://github\.com/{re.escape(owner)}/{re.escape(repo_name)}/issues/(\d+)\b",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, text):
                issue_numbers.add(int(match))

        return issue_numbers

    def store_data(self, data, owner, repo_name, pr_number, pr_details, issue_details):
        changed_code = {}
        base_code = {}
        merged_code = {}

        for key, value in data.items():
            changed_code[key] = value["changed_lines"]
            base_code[key] = value["base_content"]
            merged_code[key] = value["head_content"]

        folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{owner}_{repo_name}_pr{pr_number}",
        )
        os.makedirs(folder_path, exist_ok=True)

        with open(os.path.join(folder_path, "base_code.json"), "w", encoding="utf-8") as f:
            json.dump(base_code, f, indent=4)

        with open(os.path.join(folder_path, "merged_code.json"), "w", encoding="utf-8") as f:
            json.dump(merged_code, f, indent=4)

        with open(os.path.join(folder_path, "changed_code.json"), "w", encoding="utf-8") as f:
            json.dump(changed_code, f, indent=4)

        with open(os.path.join(folder_path, "pr_details.json"), "w", encoding="utf-8") as f:
            json.dump(pr_details, f, indent=4, ensure_ascii=False)

        with open(os.path.join(folder_path, "issue.json"), "w", encoding="utf-8") as f:
            json.dump(issue_details, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("Set GITHUB_TOKEN in your environment before running this script.")

    scraper = gitPRRetriever(token)
    state = prState(owner="pathwaycom", repo="llm-app", pr_number=0)
    scraper.run(state)

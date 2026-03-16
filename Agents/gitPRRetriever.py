from templates.state import prState
from github import Github, PullRequest, Repository
from github import Auth
import json
import os
import re


class gitPRRetriever:
    def __init__(self, token: str):
        if not token:
            raise ValueError("GitHub token must be provided.")
        auth = Auth.Token(token)
        self.g = Github(auth=auth)

        if not self.g.get_user().login:
            raise ValueError("Invalid GitHub token provided.")
        
        
        pass

    def run(self, state: prState):
        owner = state.owner
        repo_name = state.repo
        repo = None
        data = {}
        pr_details = {}
        issue_details = {"related_issue_numbers": [], "issues": []}
        pr_number = None

        try:
            repo = self.g.get_repo(f"{owner}/{repo_name}")
            print("Owner ID: ", self.g.get_user(f"{owner}").login)
            print(repo.name)
            prList = repo.get_pulls(state='closed', sort='updated', direction='desc')
            print('='*100)
            print("Latest closed PRs with less than 5 changed files: ")
            print("PR number : title")
            cnt = 0
            for pr in prList:
                if cnt >= 5:
                    break
                if(pr.changed_files < 5): # about validation of pr, we can add some rules here based on the limitations of our project.
                    cnt += 1
                    print(f"PR {pr.number}: {pr.title} , additions: {pr.additions} , deletions: {pr.deletions}, changed files: {pr.changed_files} , state: {pr.state}")
            print('='*100)

            input_pr_number = input(f"Enter the PR number that you want to review: ")
            pr_number = int(input_pr_number)
            # pr_number = 5007
            pr = repo.get_pull(pr_number)

            print(f"Selected PR : {pr_number} {pr.title}, changed files: {pr.changed_files}")

            head_ref = pr.head.sha
            base_ref = pr.base.sha
            files = pr.get_files()
            for ind, file in enumerate(files):
                print(f"File: {file.filename}, additions: {file.additions}, deletions: {file.deletions}")
                base_content = repo.get_contents(file.filename, ref=base_ref).decoded_content.decode("utf-8").strip()
                head_content = repo.get_contents(file.filename, ref=head_ref).decoded_content.decode("utf-8").strip()

                changed_lines = repo.compare(base_ref, head_ref).files[ind].patch

                data[file.filename] = {
                    "base_content": base_content,
                    "head_content": head_content,
                    "changed_lines": changed_lines
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

        except Exception as e:
            print(f"Error fetching PRs for {owner}/{repo_name}: {e}")

        try:
            if repo and pr_number is not None:
                self.storedata(data, owner, repo.name, pr_number, pr_details, issue_details)
        except Exception as e:
            print(f"Error storing data for {owner}/{repo_name}: {e}")

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

        linked_numbers = self.extract_issue_numbers("\n".join(text_blobs), repo.owner.login, repo.name)

        issue_items = []
        for number in sorted(linked_numbers):
            if number == pr.number:
                continue

            try:
                issue = repo.get_issue(number)

                # Skip linked pull requests; keep only issue artifacts.
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
            except Exception as e:
                print(f"Unable to fetch related issue #{number}: {e}")

        return {
            "related_issue_numbers": sorted(
                [item["number"] for item in issue_items]
            ),
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

    def storedata(self, data, owner, repo_name, pr_number, pr_details, issue_details):
        changed_code = {}
        base_code = {}
        merged_code = {}

        for key, value in data.items():
            changed_code[key] = value["changed_lines"]
            base_code[key] = value["base_content"]
            merged_code[key] = value["head_content"]

        folder_path = f"data/{owner}_{repo_name}_pr{pr_number}"
        os.makedirs(folder_path, exist_ok=True)

        
        with open(f"{folder_path}/base_code.json", "w", encoding="utf-8") as f:
            json.dump(base_code, f, indent=4)

        with open(f"{folder_path}/merged_code.json", "w", encoding="utf-8") as f:
            json.dump(merged_code, f, indent=4)

        with open(f"{folder_path}/changed_code.json", "w", encoding="utf-8") as f:
            json.dump(changed_code, f, indent=4)

        with open(f"{folder_path}/pr_details.json", "w", encoding="utf-8") as f:
            json.dump(pr_details, f, indent=4, ensure_ascii=False)

        with open(f"{folder_path}/issue.json", "w", encoding="utf-8") as f:
            json.dump(issue_details, f, indent=4, ensure_ascii=False)



    def validate(pr):
        if pr.state != "closed":
            return False
        if pr.changed_files >= 5:
            return False
        
        return True
        
    

        

"""
Validation Rules:
1. PR must be closed (state="closed")
2. PR addition and deletion LOC < 1000 (doesn't matter now)
3. files that are changed <5
"""




if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("Set GITHUB_TOKEN in your environment before running this script.")

    scraper = gitPRRetriever(token)
    state = prState(owner="Mintplex-Labs", repo="anything-llm", pr_number=1)
    scraper.run(state)

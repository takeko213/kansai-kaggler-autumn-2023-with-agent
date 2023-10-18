"""
githubのissueをcloseする
"""

import requests
import os
import argparse

def close_github_issue(token, repo_owner, repo_name, issue_number):
    """
    Close a GitHub issue.

    :param token: GitHub personal access token
    :param repo_owner: Repository owner (username or organization name)
    :param repo_name: Repository name
    :param issue_number: Number of the issue to be closed
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "state": "closed"
    }

    response = requests.patch(url, headers=headers, json=data)
    
    if response.status_code == 200:
        print(f"Issue #{issue_number} was closed successfully.")
    else:
        print(f"Failed to close the issue. Response: {response.content.decode()}")

parser = argparse.ArgumentParser()
parser.add_argument("--owner", required=True)
parser.add_argument("--repo", required=True)
parser.add_argument("--issue_number", required=True)
args = parser.parse_args() 


# Usage
YOUR_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = args.owner
REPO_NAME = args.repo
ISSUE_TO_CLOSE = args.issue_number

close_github_issue(YOUR_TOKEN, REPO_OWNER, REPO_NAME, ISSUE_TO_CLOSE)
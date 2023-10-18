import os
import requests
import pandas as pd


def fetch_github_issues(owner, repo):
    """
    githubのissuesの一覧を取得。pandas dataframeとして返す
    """
    token = os.getenv("GITHUB_TOKEN")

    url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=open"
    headers = {
        "Accept": "application/vnd.github.v3+json", 
        "Authorization": "token " + token
    }
    
    issues = requests.get(url, headers=headers).json()
    return pd.DataFrame(issues)


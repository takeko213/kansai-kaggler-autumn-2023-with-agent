"""
githubのissuesから実験を行うissuesを取得する
"""

import argparse 
from fetch_github_issues import fetch_github_issues

parser = argparse.ArgumentParser()
parser.add_argument("--owner", required=True)
parser.add_argument("--repo", required=True)
args = parser.parse_args() 

owner = args.owner
repo = args.repo

issues = fetch_github_issues(owner, repo)
if len(issues) > 0:
    issues = issues[issues["state"]=="open"].copy()
    issues = issues.sort_values("number")
    number = issues.iloc[0]["number"]
    title = issues.iloc[0]["title"]
else:
    number = -1
    title = ""

print(number)
print(title)
"""
wandbの実験記録から現在の最良スコアを取得する
"""

import argparse
from fetch_wandb_result import fetch_wandb_result

parser = argparse.ArgumentParser()
parser.add_argument("--score_name", required=True)
parser.add_argument("--direction", required=True)
args = parser.parse_args() 

score_name = args.score_name
direction = args.direction

result = fetch_wandb_result(score_name)
if direction == "maximize":
    best_score = result["score"].max()
elif direction == "minimize":
    best_score = result["score"].min()

print(best_score)
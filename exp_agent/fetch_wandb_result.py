import wandb
import pandas as pd
import json
import os


def fetch_wandb_result(score_name):
    """
    wandbに保存された実験結果を取得する
    """
    project = os.getenv("WANDB_PROJECT")

    api = wandb.Api()
    runs = api.runs(path=project)

    notes = []
    scores = []
    for run in runs:
        if run.state == "finished":
            notes.append(run.Notes)
            scores.append(json.loads(run.json_config)[score_name]["value"])
    df = pd.DataFrame({"Note": notes, "score":scores})
    return df


"""
wandbの最新実行runのNotesに書き込み
"""


import argparse
import wandb
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--notes", required=True)
args = parser.parse_args() 
notes = args.notes

project = os.getenv("WANDB_PROJECT")

api = wandb.Api()
runs = api.runs(path=project)
latest_run = runs[0]

if not latest_run.notes:
    print(latest_run.name, notes)
    latest_run.notes = notes
    latest_run.update()
wandb.finish()
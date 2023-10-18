import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prev", required=True)
parser.add_argument("--new", required=True)
parser.add_argument("--direction", required=True)
args = parser.parse_args() 

prev_score = args.prev
new_score = args.new
direction = args.direction

result = 0

if direction == "maximize":
    if new_score > prev_score:
        result = 1
elif direction == "minimize":
    if new_score < prev_score:
        result = 1

print(result)
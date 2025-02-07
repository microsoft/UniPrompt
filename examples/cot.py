import argparse

from uniprompt.beam_search import BeamSearch
from uniprompt.data import create_ethos_dataset, load_data
from uniprompt.evaluate import evaluate
from uniprompt.grouping import Grouping
from uniprompt.train import train
from uniprompt.utils.config_utils import load_config

parser = argparse.ArgumentParser(description="UniPrompt Implementation")
parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

print("Loading config and dataset...")
create_ethos_dataset("data/ethos.jsonl")

# Load config and dataset
config = load_config(args.config_path)
train_data, val_data, test_data = load_data(config)

# Initialize UniPrompt
beam = BeamSearch(config["beam_width"])
if "number_of_groups" in config:
    number_of_groups = config["number_of_groups"]
else:
    number_of_groups = 1

grouping = Grouping(number_of_groups)
p = config["initial_prompt"]

metrics = evaluate(data=test_data, prompt=p, config = config)
print(f"Metrics for initial prompt: {p}: {metrics}")

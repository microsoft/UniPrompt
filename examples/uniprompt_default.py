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
train_data, val_data, test_data = load_data(config["dataset_name"])

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

# Initialize candidates
beam.initialize_candidates(initial_prompt = p, data=val_data, config=config)

for epoch in range(config["epochs"]):
    # Create groups based on configured frequency
    p = beam.get_best_prompt()
    if epoch % config["group_frequency"] == 0:
        # if you want to group every epoch then you can do that or you can group based on the grouping frequency set in config
        # grouping function is optional, you can provide your own grouping function
        grouping.create_groups(prompt=p, data=train_data, config=config)

    # Training iterations
    for _ in range(config["iterations"]):
        print(f'Training start: {config["iterations"]}')
        beam = train(train_data=train_data, val_data=val_data, config=config, beam=beam, grouping = grouping)
        p = beam.get_best_prompt()
        # evaluation function is optional, you can provide your own evaluation function
        metrics = evaluate(data=test_data, prompt=p, config = config)
        print(f"Epoch: {epoch}, Prompt: {p}, Metrics: {metrics}")

# Final evaluation
p = beam.get_best_prompt()
final_metrics = evaluate(data=test_data, prompt=p, config = config)

print(f"Best prompt: {p}")
print(f"Final metrics: {final_metrics}")

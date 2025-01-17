import argparse

from uniprompt.beam_search import BeamSearch
from uniprompt.data import create_gsm8k_dataset, load_data
from uniprompt.evaluate import evaluate
from uniprompt.grouping import Grouping
from uniprompt.train import train, opro_train
from uniprompt.utils.config_utils import load_config
from uniprompt.utils.summary_utils import past_prompts_with_evaluation

parser = argparse.ArgumentParser(description="OPRO Implementation")
parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

print("Loading config and dataset...")
create_gsm8k_dataset("data/gsm8k.jsonl")

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
print(f"Metrics for initial prompt: {p}: {metrics['acc']}")

# Initialize candidates
beam.initialize_candidates(initial_prompt = p, data=val_data, config=config)

for epoch in range(config["epochs"]):

    p = past_prompts_with_evaluation(train_data, beam, config)
    print(f'Training start, epoch/step: {epoch} \n')

    prompts = []
    # Training iterations
    prompts = opro_train(p, train_data, val_data, config, beam)

    for p in prompts:
        print(f"Epoch/step: {epoch}, Prompt: {p} \n")
        metrics = beam.add_prompt_to_beam(p, val_data, config)
        print(f"Metrics: {metrics}")

# Final evaluation
p = beam.get_best_prompt()
final_metrics = evaluate(data=test_data, prompt=p, config = config)

print(f"Best prompt: {p}")
print(f"Final metrics: {final_metrics}")

import argparse
import json

from datasets import load_dataset
from uniprompt import optimize


def create_ethos_dataset(output_path):
    def write_to_jsonl(f, split, data):
        questions = data["text"]
        answers = data["label"]
        choices = ['["Non-Hate", "Hate"]'] * len(questions)
        for i in range(len(questions)):
            f.write(
                f'{{"split": "{split}", "question": "{questions[i]}", "choices": {choices[i]}, "answer": "{chr(answers[i]+65)}"}}\n'
            )

    dataset = load_dataset("ethos", "binary")["train"]

    train_ratio, val_ratio = 0.4, 0.2
    train_len = int(len(dataset) * train_ratio)
    val_len = int(len(dataset) * val_ratio)
    with open(output_path, "w") as f:
        write_to_jsonl(f, "train", dataset[:train_len])
        write_to_jsonl(f, "validation", dataset[train_len : train_len + val_len])
        write_to_jsonl(f, "test", dataset[train_len + val_len :])


def main(args):
    # Load the configuration
    config_file = args.config

    # Loading the configuration file
    with open(config_file) as f:
        config = json.load(f)

    create_ethos_dataset(output_path=config["dataset_path"])

    final_prompt = optimize(config=config)
    print(f"Optimization is done! The final prompt is: \n\n{final_prompt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read configuration from a JSON file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    main()

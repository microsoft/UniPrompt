import argparse
import json

from datasets import load_dataset
from uniprompt import UniPrompt


def create_ethos_dataset(output_path):
    def write_to_jsonl(f, split, data):
        questions = data["text"]
        answers = data["label"]
        choices = [["Non-Hate", "Hate"]] * len(questions)
        for i in range(len(questions)):
            write_data = {
                "split": split,
                "question": questions[i],
                "choices": choices[i],
                "answer": choices[i][answers[i]],
            }
            json.dump(write_data, f)
            f.write("\n")

    dataset = load_dataset(
                "ethos",
                "binary",
                trust_remote_code=True, split="train",
            ).shuffle(seed=4)


    dataset = dataset.select(range(100)) # Select 100 examples for this test run
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

    optimizer = UniPrompt(config=config)
    final_prompt = optimizer.optimize()
    print(f"Optimization is done! The final prompt is: \n\n{final_prompt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read configuration from a JSON file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    main(args)

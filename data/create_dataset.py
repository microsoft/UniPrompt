import json
import argparse
from datasets import load_dataset, concatenate_datasets
from pathlib import Path

MMLU_TASKS = ['business_ethics', 'global_facts', 'management']


def write_to_jsonl(f, split, data):
    question = data["question"]
    choices = data["choices"]
    targets = data["answer"]
    for i in range(len(question)):
        write_data = {
            "split": split,
            "question": str(question[i]) + "\n Choices: " + str(choices[i]),
            "answer": targets[i],
        }
        json.dump(write_data, f)
        f.write("\n")

def process_task(task_name, output_dir):
    splits = []
    splits.append(load_dataset("cais/mmlu", f"{task_name}", split="validation"))
    splits.append(load_dataset("cais/mmlu", f"{task_name}", split="test"))
    splits.append(load_dataset("cais/mmlu", f"{task_name}", split="dev"))
    
    # Combine all splits
    dataset = concatenate_datasets(splits)
    
    total_samples = len(dataset)
    
    output_path = output_dir / f"{task_name}.jsonl"
    
    if total_samples >= 250:
        # If dataset is large enough, take 250 samples
        dataset = dataset.shuffle(seed=42).select(range(250))
        train_len = 50
        val_len = 50
        test_len = 150
    else:
        # For smaller datasets
        dataset = dataset.shuffle(seed=42)
        train_len = 50
        val_len = 30 
        test_len = total_samples - train_len
    
    with open(output_path, "w") as f:
        write_to_jsonl(f, "train", dataset[:train_len])
        write_to_jsonl(f, "validation", dataset[train_len:train_len + val_len])
        write_to_jsonl(f, "test", dataset[train_len + val_len:])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/mmlu", help="Directory to save the dataset files")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for task in MMLU_TASKS:
        print(f"Processing task: {task}")
        process_task(task, output_dir)

if __name__ == "__main__":
    main()
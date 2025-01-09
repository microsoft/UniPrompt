import json
from typing import Optional
from datasets import load_dataset

def load_data(dataset_name: str, split: Optional[dict] = None) -> tuple:
    base_path = f"data/{dataset_name}"
    with open(f"{base_path}.jsonl") as f:
        data = [json.loads(line) for line in f]

    train_questions = []
    train_choices = []
    train_answers = []
    val_questions = []
    val_choices = []
    val_answers = []
    test_questions = []
    test_choices = []
    test_answers = []

    for question in data:
        if question["split"] == "train":
            train_questions.append(question["question"])
            train_choices.append(question["choices"])
            train_answers.append(question["answer"])
        elif question["split"] == "validation":
            val_questions.append(question["question"])
            val_choices.append(question["choices"])
            val_answers.append(question["answer"])
        elif question["split"] == "test":
            test_questions.append(question["question"])
            test_choices.append(question["choices"])
            test_answers.append(question["answer"])

    train_set = (list(train_questions), list(train_choices), list(train_answers))
    val_set = (list(val_questions), list(val_choices), list(val_answers))
    test_set = (list(test_questions), list(test_choices), list(test_answers))

    return train_set, val_set, test_set

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

def create_ethos_dataset(output_path):
    dataset = load_dataset(
                "ethos",
                "binary",
                trust_remote_code=True, split="train",
            ).shuffle(seed=4)

    dataset = dataset.select(range(100))
    train_ratio, val_ratio = 0.4, 0.2
    train_len = int(len(dataset) * train_ratio)
    val_len = int(len(dataset) * val_ratio)
    with open(output_path, "w") as f:
        write_to_jsonl(f, "train", dataset[:train_len])
        write_to_jsonl(f, "validation", dataset[train_len : train_len + val_len])
        write_to_jsonl(f, "test", dataset[train_len + val_len :])

import json
from typing import Optional


def load_dataset(dataset_name: str, split: Optional[dict] = None) -> tuple:
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

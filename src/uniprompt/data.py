import json
from typing import Optional
from datasets import load_dataset
from uniprompt.utils.summary_utils import load_prompts
import re
from uniprompt.utils.api_utils import chat_completion

def load_data(dataset_name: str, split: Optional[dict] = None) -> tuple:
    base_path = config["dataset_name"]
    with open(f"{base_path}.jsonl") as f:
        data = [json.loads(line) for line in f]

    train_questions = []
    train_answers = []
    val_questions = []
    val_answers = []
    test_questions = []
    test_answers = []

    for question in data:
        if question["split"] == "train":
            train_questions.append(question["question"])
            train_answers.append(question["answer"])
        elif question["split"] == "validation":
            val_questions.append(question["question"])
            val_answers.append(question["answer"])
        elif question["split"] == "test":
            test_questions.append(question["question"])
            test_answers.append(question["answer"])

    train_set = (list(train_questions), list(train_answers))
    val_set = (list(val_questions), list(val_answers))
    test_set = (list(test_questions), list(test_answers))

    return train_set, val_set, test_set

def write_to_jsonl(f, split, data):
    questions = data["text"]
    answers = data["label"]
    for i in range(len(questions)):
        write_data = {
            "split": split,
            "question": questions[i],
            "answer": f"{answers[i]}",
        }
        json.dump(write_data, f)
        f.write("\n")

def create_ethos_dataset(output_path):
    dataset = load_dataset(
                "ethos",
                "binary",
                trust_remote_code=True, split="train",
            ).shuffle(seed=4)

    dataset = dataset.select(range(250))
    train_len = 50
    val_len = 50
    with open(output_path, "w") as f:
        write_to_jsonl(f, "train", dataset[:train_len])
        write_to_jsonl(f, "validation", dataset[train_len:train_len + val_len])
        write_to_jsonl(f, "test", dataset[train_len + val_len:])

def default_write_to_jsonl(f, split, data):
    questions = data["question"]
    answers = data["answer"]
    for i in range(len(questions)):
        # Extract the answer after '####'
        extracted_answer = re.search(r'####\s*(.*)', answers[i]).group(1).strip()
        
        write_data = {
            "split": split,
            "question": questions[i],
            "answer": extracted_answer,
        }
        json.dump(write_data, f)
        f.write("\n")

def create_gsm8k_dataset(output_path):
    train_dataset = load_dataset(
                "openai/gsm8k",
                "main",
                trust_remote_code=True, split="train",
            ).shuffle(seed=4)

    test_dataset = load_dataset(
                "openai/gsm8k",
                "main",
                trust_remote_code=True, split="test",
            ).shuffle(seed=4)

    train_len = 300
    val_len = 50
    test_len = 100
    train_subset = train_dataset.select(range(train_len))
    val_subset = train_dataset.select(range(train_len, train_len + val_len))
    test_subset = test_dataset.select(range(test_len))

    with open(output_path, "w") as f:
        default_write_to_jsonl(f, "train", train_subset)
        default_write_to_jsonl(f, "validation", val_subset)
        default_write_to_jsonl(f, "test", test_subset)

def add_rationale_to_dataset(dataset_name: str, config):
    prompts = load_prompts()
    add_rationale_prompt = prompts.get("add_rationale", None)

    base_path = f"data/{dataset_name}"
    dataset_path = f"{base_path}.jsonl"
    output_path = f"{base_path}_rationale.jsonl"

    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    for question in data:
        add_rationale_prompt = add_rationale_prompt.format(question=question["question"], answer=question["answer"])
        messages = [{"role": "user", "content": add_rationale_prompt}]
        rationale = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
        question["rationale"] = rationale

    with open(output_path, "w") as f:
        for question in data:
            json.dump(question, f)
            f.write("\n")
import copy
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Sequence, Tuple, Union

import msal
import numpy as np
import pkg_resources
import requests
import yaml
from openai import AzureOpenAI, OpenAI



def load_prompts() -> Dict[str, str]:
    prompt_path = pkg_resources.resource_filename("uniprompt", os.path.join("prompts", "default.yaml"))
    with open(prompt_path) as f:
        prompts = yaml.safe_load(f)
    return prompts

def get_confusion_matrix(y_true: Sequence[Any], y_pred: Sequence[Any], normalize: bool = False) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    classes.sort()
    n_classes = len(classes)

    cm = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))

    if normalize is True:
        for i in range(n_classes):
            if np.sum(cm[i, :]) == 0:
                cm[i, :] = 0
            else:
                cm[i, :] = cm[i, :]/np.sum(cm[i, :])

    return cm

def chat_completion(cache_path=None, papyrus=False, **kwargs):
    def make_api_call(papyrus=False, **kwargs):
        if papyrus is True:
            from uniprompt.papyrus import chat_completion_papyrus
            return chat_completion_papyrus(cache_path=cache_path, **kwargs)

        else: # OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")

            if kwargs["api_kwargs"]["api_type"] == "azure":
                client = AzureOpenAI(api_key=api_key, azure_endpoint=kwargs["api_kwargs"]["api_base"], api_version=kwargs["api_kwargs"]["api_version"])
            else:
                client = OpenAI(api_key=api_key)
                # breakpoint()
            while True:
                try:
                    
                    response = client.chat.completions.create(**kwargs["model_kwargs"], messages=kwargs["messages"])
                    print(response)
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Got error {e}. Sleeping for 5 seconds...")
                    time.sleep(5)      

    if not cache_path:
        return make_api_call(papyrus=papyrus, **kwargs)

    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)

    # Connect to SQLite database
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_cache
    (model TEXT, messages TEXT, response TEXT)
    """)

    # Create a unique key from model and messages
    model = json.dumps(kwargs["model_kwargs"])
    messages = json.dumps(kwargs["messages"])

    # Check if the response is in the cache
    cursor.execute("SELECT response FROM api_cache WHERE model = ? AND messages = ?", (model, messages))
    cached_response = cursor.fetchone()

    if cached_response:
        conn.close()
        return cached_response[0]

    # If not in cache, make the API call
    response_content = make_api_call(papyrus=papyrus, **kwargs)

    # Store the response in the cache
    cursor.execute("INSERT INTO api_cache (model, messages, response) VALUES (?, ?, ?)",
                   (model, messages, response_content))
    conn.commit()
    conn.close()

    return response_content

def load_dataset_code(dataset_path):
    from datasets import load_dataset
    data = load_dataset("openai_humaneval")
    qustions = []
    tests = []
    choices = []
    for i in range(len(data['test']['task_id'])):
        prompt = data['test']['prompt'][i]
        test = data['test']['test'][i]
        entry_point = data['test']['entry_point'][i]
        soln = data['test']['canonical_solution'][i]
        qustions.append(prompt)
        tests.append(test + "\n" + f"check({entry_point})")
        choices.append(soln)
    
    dataset_dict = {
        "train_questions": qustions,
        "train_answers": tests,
        "train_choices": choices, 
        "val_questions": qustions[100:150],
        "val_answers": tests[100:150],
        "val_choices": choices[100:150],
        "test_questions": qustions[150:],
        "test_answers": tests[150:],
        "test_choices": choices[150:],
    }
    return dataset_dict
    

def load_dataset(dataset_path: str) -> Dict[str, List[Union[str, List[str]]]]:
    """
    Reads the dataset from a jsonl file where the dataset is stored in the following format
    split, question, choices, answer
    Returns the questions, choices and answers as lists

    Args:
        dataset_path (str): Path to the dataset file
    Returns:
        A dictionary containing the train, val and test questions, choices and answers
    """

    with open(dataset_path) as f:
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

    dataset_dict = {
        "train_questions": train_questions,
        "train_answers": train_answers,
        "train_choices": train_choices,
        "val_questions": val_questions,
        "val_answers": val_answers,
        "val_choices": val_choices,
        "test_questions": test_questions,
        "test_answers": test_answers,
        "test_choices": test_choices,
    }
    return dataset_dict


def feedback_one_example(
    prompt: str,
    prompt_template: str,
    questions: Sequence[str],
    answers: Sequence[str],
    pred_answers: Sequence[str],
    cots: Sequence[str],
    config: Dict[str, Any]
) -> str:
    """
    Given a prompt, questions, answers, predicted answers and explanations, provide feedback to the student.

    Args:
        prompt: The prompt given to the student.
        questions: The questions given to the student.
        answers: The correct answers to the questions.
        pred_answers: The answers given by the student.
        cots: The explanations given by the student.
        config: The configuration for the feedback model.

    Returns:
        The feedback given to the student.
    """

    examples = ""
    for i in range(len(pred_answers)):
        examples += f"### Question\n{questions[i]}### Answer\n{answers[i]}### Predicted answer\n{pred_answers[i]}### Explanation\n{cots[i]}\n\n"

    input_prompt = prompt_template.format(prompt=prompt, examples=examples)
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(cache_path=config["cache_path"], papyrus=config["papyrus"], **config["expert_llm"], messages=messages)
    return output


def apply_edits(prompt: str, prompt_template: str, edits: str, config: Dict[str, Any]) -> str:
    """
    Apply the edits to the prompt and return the final prompt.

    Args:
        prompt: The prompt to which the edits are to be applied.
        edits: The edits to be applied.
        config: The configuration for the feedback model.

    Returns:
        The final prompt after applying the edits.
    """

    input_prompt = prompt_template.format(prompt=prompt, edits=edits)
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(cache_path=config["cache_path"], papyrus=config["papyrus"], **config["expert_llm"], messages=messages)
    return output


def feedback_with_history(
    prompt: str,
    prompt_template: str,
    questions: Sequence[str],
    answers: Sequence[str],
    pred_answers: Sequence[str],
    cots: Sequence[str],
    history: List[Tuple[str, float]],
    config: Dict[str, Any]
) -> str:
    """
    Given a history of feedbacks, provide a single line feedback to the student.

    Args:
        prompt: The prompt given to the student.
        questions: The questions given to the student.
        answers: The correct answers to the questions.
        pred_answers: The answers given by the student.
        cots: The explanations given by the student.
        history: The history of feedbacks given to the student.
        config: The configuration for the feedback model.

    Returns:
        The feedback given to the student.
    """

    examples = ""
    for i in range(len(pred_answers)):
        examples += f"""
### Question
    {questions[i]}
### True Answer
    {answers[i]}
### Student's answer
    {pred_answers[i]}
### Explanation
    {cots[i]}
"""

    history_string = ""
    for i in range(len(history)):
        history_string += f"""
### Edit Proposed
    {history[i][0]}
### Accuracy Change
    {history[i][1]}
"""

    input_prompt = prompt_template.format(prompt=prompt, examples=examples, history_string=history_string)
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(cache_path=config["cache_path"], papyrus=config["papyrus"], **config["expert_llm"], messages=messages)
    return output


def combine_multiple_feedbacks_with_examples(
    prompt_template: str,
    edits: str,
    wrong_examples: Sequence[str],
    config: Dict[str, Any]
) -> str:
    """
    Combine multiple feedbacks into a summary and provide edits to the prompt.

    Args:
        edits: The edits to be applied to the prompt.
        wrong_examples: The examples of wrong questions.
        config: The configuration for the feedback model.

    Returns:
        The summary of the feedbacks.
    """

    wrong_examples_string = ""
    for i in range(len(wrong_examples)):
        wrong_examples_string += f"\n### Question{wrong_examples[i]}\n"

    input_prompt = prompt_template.format(edits=edits, wrong_examples_string=wrong_examples_string)
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(cache_path=config["cache_path"], papyrus=config["papyrus"], **config["expert_llm"], messages=messages)
    return output

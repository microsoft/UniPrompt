import json
import os
import re
from typing import Dict, List, Sequence

from .utils import (
    apply_edits,
    chat_completion,
    combine_multiple_feedbacks_with_examples,
    feedback_one_example,
    feedback_with_history,
    load_dataset,
)


def extract_answer(cot: str) -> str:
    """Extracts the answer from the chain of thought answer."""
    pattern = r"<Answer>(.*?)<\/Answer>"
    match = re.search(pattern, cot)
    if match:
        return match.group(1)
    return None


def make_prompt(prompt: str, question: str, choices: Sequence[List[str]]) -> str:
    question_str = f"Question: {question}\n"
    output_format_str = ""
    cot_str = "Give an explanation before the final answer."
    if len(choices) > 0:
        answer_string = ""
        question_str += "ANSWER CHOICES: "
        for i in range(len(choices)):
            question_str += f"{chr(65+i)}:{choices[i]}, "
            answer_string += "<Answer>" + chr(65 + i) + "</Answer> or"
        output_format_str = f"output_format: Give your answer as the correct option between <Answer></Answer> tags like: {answer_string[:-2]}."
    else:
        output_format_str = "output_format: Give your answer as the correct number between <Answer></Answer> tags like: <Answer>201</Answer> or <Answer>5505</Answer> or <Answer>1</Answer> etc."

    made_prompt = f"{prompt}\n{question_str}\n{output_format_str} {cot_str}"
    return made_prompt


def evaluate_prompt(
    new_prompt: str,
    questions: Sequence[str],
    answers: Sequence[str],
    choices: Sequence[List[str]],
    config: Dict,
) -> float:
    """
    Evaluates the prompt on a set of questions, choices and answers.
    Returns the accuracy of the prompt on the given set of questions.

    Args:
        new_prompt (str): The prompt to evaluate
        questions (Sequence[str]): The list of questions
        answers (Sequence[str]): The list of answers
        choices (Sequence[List[str]]): The list of choices
    Returns:
        The accuracy of the prompt on the given set of questions.
    """
    acc = 0

    for question, answer, choice in zip(questions, answers, choices):
        prompt = make_prompt(prompt=new_prompt, question=question, choices=choice)
        messages = [{"role": "user", "content": prompt}]
        answer_cot = chat_completion(**config["solver_llm"], messages=messages)
        answer_llm = extract_answer(answer_cot)
        if str(answer) in str(answer_llm) or str(answer_llm) in str(answer):
            acc += 1 / len(questions)

    return acc


def log_iteration(iteration_data, file_path):
    with open(file_path, "a") as json_file:
        json.dump(iteration_data, json_file)
        json_file.write("\n")


def make_groups(
    initial_prompt: str,
    questions: Sequence[str],
    answers: Sequence[str],
    choices: Sequence[List[str]],
    config: Dict,
) -> Dict[int, List[int]]:
    """
    Groups the questions into clusters based on the feedbacks.

    Args:
        initial_prompt (str): The initial prompt
        questions (Sequence[str]): The list of questions
        answers (Sequence[str]): The list of answers
        choices (Sequence[List[str]]): The list of choices
    Returns:
        A dictionary with the cluster number as the key and the list of indices of the questions in that cluster as the value.
    """
    feedbacks = []
    for question, choice, answer in zip(questions, choices, answers):
        prompt = make_prompt(prompt=initial_prompt, question=question, choices=choice)
        messages = [{"role": "user", "content": prompt}]
        answer_cot = chat_completion(**config["solver_llm"], messages=messages)
        answer_llm = extract_answer(answer_cot)
        if answer_llm is not None and str(answer_llm) != str(answer):
            wrong_choices = choice[ord(answer_llm) - 65]
            wrong_cots = answer_cot
            correct_choices = choice[ord(answer) - 65]
            question_feedback = feedback_one_example(
                prompt=initial_prompt,
                questions=[question],
                answers=[correct_choices],
                pred_answers=[wrong_choices],
                cots=[wrong_cots],
                config=config,
            )
            feedbacks.append(question_feedback)
        else:
            feedbacks.append("Correct Answer")

    feedback = "\n Feedback: ".join(feedbacks)

    prompt = f"""You are given a set of feedbacks, you need to cluster them into five groups based on similarity, and then provide a summary of each group. You can use the following feedbacks to cluster: \n {feedback}

Provide each cluster explanation within the following tags: <Cluster></Cluster>"""

    messages = [{"role": "user", "content": prompt}]
    cluster = chat_completion(**config["grouping_llm"], messages=messages)
    clusters = re.findall(r"<Cluster>(.*?)</Cluster>", cluster, re.DOTALL)

    groups = {}
    groups[0] = []
    string_of_clusters = "Group 0: Correct Answer \n"
    i = 1
    for cluster in clusters:
        groups[i] = []
        string_of_clusters += f"Group {i}: {cluster} \n"
        i += 1

    i = 0
    for question, choice, answer, feedback in zip(questions, choices, answers, feedbacks):
        prompt = f"""You are given a feedback and a set of clusters, you need to tell which cluster this feedback belongs to.

The clusters are: \n {string_of_clusters}

The feedback is: {feedback}

give your final answer as the number of the correct cluster between <Answer></Answer> tags like: <Answer>1</Answer>."""
        messages = [{"role": "user", "content": prompt}]
        cluster_number = chat_completion(**config["grouping_llm"], messages=messages)
        cluster_number_extracted = re.search(r"<Answer>(.*?)</Answer>", cluster_number)
        groups[int(cluster_number_extracted.group(1))].append(i)
        i += 1

    return groups


def optimize(
    config: Dict,
) -> str:

    logging_file_path = config["logging_file_path"]
    initial_prompt = config["initial_prompt"]
    dataset_path = config["dataset_path"]

    # Hyperparameters
    mini_batch = config["mini_batch"]
    batch_size = config["batch_size"]
    iterations = config["iterations"]
    epochs = config["epochs"]

    if not os.path.exists(logging_file_path):
        os.makedirs(logging_file_path, exist_ok=True)

    dataset_dict = load_dataset(dataset_path)
    train_questions, train_choices, train_answers = (
        dataset_dict["train_questions"],
        dataset_dict["train_choices"],
        dataset_dict["train_answers"],
    )
    val_questions, val_choices, val_answers = (
        dataset_dict["val_questions"],
        dataset_dict["val_choices"],
        dataset_dict["val_answers"],
    )
    test_questions, test_choices, test_answers = (
        dataset_dict["test_questions"],
        dataset_dict["test_choices"],
        dataset_dict["test_answers"],
    )

    # Clustering train set based on feedback
    groups = make_groups(
        initial_prompt=initial_prompt,
        questions=train_questions,
        answers=train_answers,
        choices=train_choices,
        config=config,
    )

    # Initializing the lists to store accuracies
    accuracies_val = []
    accuracies_test = []
    accuracies_train = []
    training_step = []
    logging_information = []

    # Initializing the beam
    prompt_0 = initial_prompt
    prompt_1 = initial_prompt
    prompt_2 = initial_prompt

    # Initial evaluation
    accuracies_val.append(
        evaluate_prompt(
            new_prompt=initial_prompt,
            questions=val_questions,
            answers=val_answers,
            choices=val_choices,
            config=config,
        )
    )
    accuracies_train.append(
        evaluate_prompt(
            new_prompt=initial_prompt,
            questions=train_questions,
            answers=train_answers,
            choices=train_choices,
            config=config,
        )
    )
    accuracies_test.append(
        evaluate_prompt(
            new_prompt=initial_prompt,
            questions=test_questions,
            answers=test_answers,
            choices=test_choices,
            config=config,
        )
    )
    training_step.append(0)

    logging_information = {
        "K-fold": 0,
        "epoch": "-1",
        "group": "-1",
        "accuracies_test": accuracies_test[-1],
        "accuracies_val": accuracies_val[-1],
        "accuracies_train": accuracies_train[-1],
        "training_step": training_step[-1],
        "prompt": prompt_1,
    }
    log_iteration(logging_information, logging_file_path)

    for epoch in range(epochs):
        # Initializing edits history
        edit_history_dict = {}
        for group in groups:
            edit_history_dict[group] = []

        # Can go over the dataset for multiple iterations
        for _ in range(iterations):
            for group in groups:

                # Some initializations
                total = 0
                acc_batch = 0
                feedback = ""
                selected_indices = groups[group]
                batch_questions = [train_questions[index] for index in selected_indices]
                batch_answers = [train_answers[index] for index in selected_indices]
                batch_choices = [train_choices[index] for index in selected_indices]

                # Going over the batch
                for z_ in range(batch_size):

                    # Some initializations
                    mini_batch_questions = batch_questions[mini_batch * z_ : mini_batch * (z_ + 1)]
                    mini_batch_answers = batch_answers[mini_batch * z_ : mini_batch * (z_ + 1)]
                    mini_batch_choices = batch_choices[mini_batch * z_ : mini_batch * (z_ + 1)]
                    wrong_questions = []
                    wrong_choices = []
                    wrong_cots = []
                    correct_choices = []

                    # Identifying failure cases of the current prompt
                    for question, answer, choices in zip(mini_batch_questions, mini_batch_answers, mini_batch_choices):
                        done = 0
                        while done == 0:
                            try:
                                prompt = make_prompt(prompt=prompt_1, question=question, choices=choices)
                                messages = [{"role": "user", "content": prompt}]
                                answer_cot = chat_completion(
                                    **config["solver_llm"],
                                    messages=messages,
                                )
                                answer_llm = extract_answer(answer_cot)
                                if str(answer_llm) == str(answer):
                                    acc_batch += 1 / len(batch_questions)
                                else:
                                    wrong_choices.append(choices[ord(answer_llm) - 65])
                                    wrong_cots.append(answer_cot)
                                    wrong_questions.append(question)
                                    correct_choices.append(choices[ord(answer) - 65])
                                total += 1
                                done = 1
                            except:
                                continue

                    # Providing feedback for the failure cases
                    if len(wrong_questions) > 0:
                        feedback_new = feedback_with_history(
                            prompt=prompt_1,
                            questions=wrong_questions,
                            answers=correct_choices,
                            pred_answers=wrong_choices,
                            cots=wrong_cots,
                            history=edit_history_dict[group],
                            config=config,
                        )
                        feedback += feedback_new + "====================="

                # Combining feedbacks over mini-batches
                if feedback != "":
                    final_feedback = combine_multiple_feedbacks_with_examples(
                        edits=feedback, wrong_examples=wrong_questions, config=config
                    )

                    # Applying edits to the beam
                    prompt_2 = apply_edits(prompt=prompt_1, edits=final_feedback, config=config)
                    prompt_3 = apply_edits(prompt=prompt_0, edits=final_feedback, config=config)

                    # Evaluating the new prompts
                    acc_0 = evaluate_prompt(
                        new_prompt=prompt_0,
                        questions=batch_questions,
                        answers=batch_answers,
                        choices=batch_choices,
                        config=config,
                    )
                    acc_1 = evaluate_prompt(
                        new_prompt=prompt_1,
                        questions=batch_questions,
                        answers=batch_answers,
                        choices=batch_choices,
                        config=config,
                    )
                    acc_2 = evaluate_prompt(
                        new_prompt=prompt_2,
                        questions=batch_questions,
                        answers=batch_answers,
                        choices=batch_choices,
                        config=config,
                    )
                    acc_3 = evaluate_prompt(
                        new_prompt=prompt_3,
                        questions=batch_questions,
                        answers=batch_answers,
                        choices=batch_choices,
                        config=config,
                    )

                    # Selecting the best prompts
                    text_number_pairs = list(
                        zip(
                            [prompt_0, prompt_1, prompt_2, prompt_3],
                            [acc_0, acc_1, acc_2, acc_3],
                        )
                    )
                    sorted_pairs = sorted(text_number_pairs, key=lambda x: x[1], reverse=True)
                    top_pair1, top_pair2 = sorted_pairs[:2]
                    prompt_1, prompt_0 = top_pair1[0], top_pair2[0]
                    acc_top, acc_sec_top = top_pair1[1], top_pair2[1]

                    # Evaluating the best prompt
                    accuracies_val.append(
                        evaluate_prompt(
                            new_prompt=prompt_1,
                            questions=val_questions,
                            answers=val_answers,
                            choices=val_choices,
                            config=config,
                        )
                    )
                    accuracies_test.append(
                        evaluate_prompt(
                            new_prompt=prompt_1,
                            questions=test_questions,
                            answers=test_answers,
                            choices=test_choices,
                            config=config,
                        )
                    )
                    accuracies_train.append(
                        evaluate_prompt(
                            new_prompt=prompt_1,
                            questions=train_questions,
                            answers=train_answers,
                            choices=train_choices,
                            config=config,
                        )
                    )
                    training_step.append(training_step[-1] + 1)

                    # Logging the information
                    logging_information = {
                        "K-fold": 0,
                        "epoch": epoch,
                        "group": group,
                        "accuracies_test": accuracies_test[-1],
                        "accuracies_val": accuracies_val[-1],
                        "accuracies_train": accuracies_train[-1],
                        "training_step": training_step[-1],
                        "prompt": prompt_1,
                    }
                    log_iteration(logging_information, logging_file_path)

                    with open(logging_file_path, "a") as json_file:
                        json_file.write("\n\n")

                    # Updating edit history
                    edit_history_dict[group].append([final_feedback, acc_top - acc_sec_top])
    return prompt_1

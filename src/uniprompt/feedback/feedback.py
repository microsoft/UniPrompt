from typing import Any, Dict, Sequence

from uniprompt.utils.api_utils import chat_completion
from uniprompt.utils.summary_utils import load_prompts
from uniprompt.evaluate import evaluate
from uniprompt.utils.sampling_utils import random_batch
import numpy as np
import random

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
        examples += f"### Question\n{questions[i]}\n### Answer\n{answers[i]}\n### Predicted answer\n{pred_answers[i]}\n### Explanation\n{cots[i]}\n\n"

    input_prompt = prompt_template.format(prompt=prompt, examples=examples)
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
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
    output = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
    return output


def feedback_with_history(
    prompt: str,
    prompt_template: str,
    questions: Sequence[str],
    answers: Sequence[str],
    pred_answers: Sequence[str],
    cots: Sequence[str],
    history: str,
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

    input_prompt = prompt_template.format(prompt=prompt, examples=examples, history_string=history)
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
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
    output = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
    return output

def generate_gradeints(prompt: str, wrong_examples: Sequence[str], num_feedbacks: int, config: Dict[str, Any]) -> str:
    error_string = ""
    for i in range(len(wrong_examples)):
        error_string += f"\n### Question{wrong_examples[i]}\n"

    prompts = load_prompts()
    generate_gradeints_prompt = prompts.get("generate_gradeints", None)
    input_prompt = generate_gradeints_prompt.format(prompt=prompt, error_string=error_string, num_feedbacks=num_feedbacks)
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
    return output

def gradient_feedback(prompt: str, wrong_examples: Sequence[str], gradient: str, steps_per_gradient: int, config: Dict[str, Any]) -> str:
    prompts = load_prompts()
    gradient_feedback_prompt = prompts.get("gradient_feedback", None)
    error_str = ""
    for i in range(len(wrong_examples)):
        error_str += f"\n### Question{wrong_examples[i]}\n"

    input_prompt = gradient_feedback_prompt.format(prompt=prompt, error_str=error_str, gradient = gradient, steps_per_gradient = steps_per_gradient)
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
    return output

def semantic_similar_prompts(prompt_instruction: str, config: Dict[str, Any]) -> str:
    prompts = load_prompts()
    semantic_similar_prompts_prompt = prompts.get("semantic_similar_prompts", None)
    input_prompt = semantic_similar_prompts_prompt.format(prompt_instruction=prompt_instruction)
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
    return output

def ucb_select(prompts, dataset, T, config, c=1.0):
    n = len(prompts)
    N_t = np.zeros(n)
    Q_t = np.zeros(n)

    for t in range(1, T + 1):
        D_sample = random_batch(dataset, min(len(dataset), 10))

        ucb_values = Q_t + c * np.sqrt(np.log(t) / (N_t + 1e-5))
        p_i = np.argmax(ucb_values)
        r_i_t = evaluate(prompt = prompts[p_i], dataset = D_sample, config = config)

        N_t[p_i] += len(D_sample)
        Q_t[p_i] = Q_t[p_i] + (r_i_t / N_t[p_i])

    top_indices = np.argsort(Q_t)[-beam_width:][::-1]
    top_prompts = [prompts[i] for i in top_indices]

    return top_prompts
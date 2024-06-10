import json
from typing import Dict, Sequence

from openai import OpenAI


def chat_completion(**kwargs):
    client = OpenAI()
    return client.chat.completions.create(**kwargs)


def load_dataset(dataset_path: str) -> Dict:
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
    questions: Sequence[str],
    answers: Sequence[str],
    pred_answers: Sequence[str],
    cots: Sequence[str],
    config: Dict,
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

    Examples = ""
    for i in range(len(pred_answers)):
        Examples += f"### Question\n{questions[i]}### Answer\n{answers[i]}### Predicted answer\n{pred_answers[i]}### Explanation\n{cots[i]}\n\n"

    input_prompt = f"""You are a teacher and you have to give feedback to your students on their answers.

You are teaching hate speach detection to your students. You are given a question and it's answer. You are also given the explanations written by your students while solving the questions.

The questions are answered wrong by the students. You have to tell why is the solution wrong and what information is can be added to the in the Background Knowledge part that would have helped the student to write better explanations.

Be explicit and tell the exact information that can be dded without further modification / addition.

You can  add a section, add a subsection, set the content of a section, set the content of a subsection, delete a section or delete a subsection in the background knowledge part.

Give very granular feedbacks, like if the student has made a mistake in the calculation, then tell what is the mistake in the calculation and how to correct it, if the student has made a mistake in the concept, then tell what is the mistake in the concept and how to correct it.

You can also give examples to make the concept more clear.

## Background Knowledge
{prompt}

{Examples}

Now, it is your turn to give feedbacks to the students.
"""

    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(**config["expert_llm"], messages=messages)
    return output


def apply_edits(prompt: str, edits: str, config: Dict) -> str:
    """
    Apply the edits to the prompt and return the final prompt.

    Args:
        prompt: The prompt to which the edits are to be applied.
        edits: The edits to be applied.
        config: The configuration for the feedback model.

    Returns:
        The final prompt after applying the edits.
    """

    input_prompt = f"""You are given an input prompt and a feedback, you have to incorporate the feedback into the input prompt and output the final prompt.
An example of the task is given below

### Input Prompt
Introduction: In this task you have to answer the given question.

### Feedback
The background knowledge is incomplete, it does not include what are the factors that affect the water usage and how many water sources are there.
\\add_subsection("Background Knowledge")
\\add_subsection_content(water usage depends on the population, climate, economic development, and availability of water sources. There are two sources of water, surface water and groundwater.)

### Final Prompt
Introduction: In this task you have to answer the given question.
Background Knowledge: water usage depends on the population, climate, economic development, and availability of water sources. There are two sources of water, surface water and groundwater.

Only output the final prompt nothing else.

### INPUT PROMPT
{prompt}

### FEEDBACK
{edits}


### FINAL PROMPT
"""
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(**config["expert_llm"], messages=messages)
    return output


def feedback_with_history(
    prompt: str, questions: Sequence[str], answers: Sequence[str], pred_answers: Sequence[str], cots, history, config
):
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

    Examples = ""
    for i in range(len(pred_answers)):
        Examples += f"""
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
    input_prompt = f"""You are a teacher and you have to give feedback to your students on their answers.

You are teaching how to solve math problems to your students. You are given a question, it's true answer and answer given by student. You are also given the explanations written by your students while solving the questions.

The questions are answered wrong by the students. You have to tell why is the solution wrong and what information is can be added to the in the Background Knowledge part that would have helped the student to write better explanations.

## IMPORTANT: You are also given a history of changes you made to the background knowledge part and the change in student's accuracy after making the change. You have to use this history to make your feedback.

Be explicit and tell the exact information that can be added without further modification / addition.

### IMPORTANT: Give feedback in form of instructions like  add a section, add a subsection, set the content of a section, set the content of a subsection, delete a section or delete a subsection in the background knowledge part.

Give very granular feedbacks, like if the student has made a mistake in the calculation, then tell what is the mistake in the calculation and how to correct it, if the student has made a mistake in the concept, then tell what is the mistake in the concept and how to correct it.

## Background Knowledge
{prompt}

## History
{history_string}


{Examples}

Now, it is your turn to give feedbacks to the students.
You can only provide a one line feedback.
"""

    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(**config["expert_llm"], messages=messages)
    return output


def combine_multiple_feedbacks_with_examples(edits, wrong_examples, config):
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

    input_prompt = f"""You are given a set of feedbacks for some problems. The set feedbacks for each problem separated by =========== symbol.
You have to summarize the feedbacks into a final feedback.
You are also given a set of wrong questions. You need to tell which edit can be applied to aid the student in solving the wrong question.

To achieve your task, try to follow the following steps;
1. Identify the general problem that is being solved by all the feedbacks.
2. Once you have identified the problem, try to make a new feedback that covers most of the feedbacks given. Let's say the problem in the first feedback is the absence of methods to solve linear equation and in the second feedback it is the method to inverse a matrix. You know that both of these problems can be caused by adding how to solve convert a matrix into row rediced echolon form. So, add that.
3. Try and validate your feedback. Once, you have a feedback try to see if it covers every feedback, if it does not cover any feedback, add that to your new feedback.
4. See the wrong questions and try to identify what is the problem in the question. If the problem is not covered by your feedback, add that to your feedback.
5. You can add specifics like examples, definitions etc make sure that the feedback is enough to be directly added without any modification.

You may use the following function templates-

add_section(sectioname)
add_subsection(section_name, subsection_name)
set_section_content(section_name, new_content)
set_subsection_content(section_name, subsection_name, new_content)
delete_section(section_name)
delete_subsection(section_name, subsection_name)

Your summary cannot include more than four functions. Make sure that the content is useful, not just a very general statement. Something specific.

Instructions:
{edits}

Wrong Questions:
{wrong_examples_string}

Summary:
"""
    messages = [{"role": "user", "content": input_prompt}]
    output = chat_completion(**config["expert_llm"], messages=messages)
    return output

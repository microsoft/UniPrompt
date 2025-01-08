import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from uniprompt.feedback.feedback import feedback_one_example
from uniprompt.utils.api_utils import chat_completion, extract_answer
from uniprompt.utils.prompt_utils import load_prompts, make_prompt


class Grouping:
    def __init__(self, number_of_groups: int):
        self.number_of_groups = number_of_groups
        self.groups = {i: [] for i in range(number_of_groups+1)}
        self.edit_history_dict = {group: [] for group in self.groups}

    def create_groups(self, prompt: str, data: Tuple, config: Dict, grouping_fn: Optional[Callable] = None) -> List[List]:
        train_questions, train_choices, train_answers = data

        if grouping_fn is None:
            grouping_fn = default_grouping_fn

        self.groups = grouping_fn(prompt, train_questions, train_choices, train_answers, config)

def default_grouping_fn(
    prompt: str,
    questions: Sequence[str],
    answers: Sequence[str],
    choices: Sequence[List[str]],
    config: Dict[str, Any]
) -> Dict[int, List[int]]:
    feedbacks = []
    prompts = load_prompts()
    for question, choice, answer in zip(questions, choices, answers):
        formatted_prompt = make_prompt(prompt=prompt, question=question, choices=choice)
        messages = [{"role": "user", "content": formatted_prompt}]
        answer_cot = chat_completion(cache_path=config["cache_path"], **config["solver_llm"], messages=messages)
        answer_llm = extract_answer(answer_cot)
        if answer_llm is not None and str(answer_llm) != str(answer):
            wrong_choices = answer_llm
            wrong_cots = answer_cot
            correct_choices = answer
            question_feedback = feedback_one_example(
                prompt=formatted_prompt,
                prompt_template=prompts.get("feedback_one_example", None),
                questions=[question],
                answers=[correct_choices],
                pred_answers=[wrong_choices],
                cots=[wrong_cots],
                config=config,
            )
            feedbacks.append(question_feedback)
        else:
            feedbacks.append("Correct Answer")

    group_feedbacks = "\nFeedback: ".join(feedbacks)
    group_prompt = prompts.get("group_prompt", None).format(feedbacks=group_feedbacks, number_of_groups = config["number_of_groups"])
    messages = [{"role": "user", "content": group_prompt}]
    all_groups = re.findall(r"<Cluster>(.*?)</Cluster>", chat_completion(cache_path=config["cache_path"], **config["grouping_llm"], messages=messages), re.DOTALL)

    groups = {}
    groups[0] = []
    groups_str = "Group 0: Correct Answer\n"
    for i, g in enumerate(all_groups):
        groups[i+1] = [] # i+1 because 0 is reserved for correct answers
        groups_str += f"Group {i}: {g}\n"

    for idx, (question, choice, answer, feedback) in enumerate(zip(questions, choices, answers, feedbacks)):
        assign_group_prompt = prompts.get("assign_group_prompt", None).format(groups_str=groups_str, feedback=feedback, number_of_groups = config["number_of_groups"])
        messages = [{"role": "user", "content": assign_group_prompt}]
        cluster_number = chat_completion(cache_path=config["cache_path"], **config["grouping_llm"], messages=messages)

        cluster_number_extracted = int(re.search(r"<Answer>(.*?)</Answer>", cluster_number).group(1))
        groups[cluster_number_extracted].append(idx)

    print(f"Groups: {groups}")

    return groups

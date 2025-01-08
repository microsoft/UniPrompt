from typing import Any, Dict, List, Sequence, Union

from uniprompt.utils.api_utils import chat_completion, extract_answer
from uniprompt.utils.metric_utils import get_confusion_matrix, get_metric
from uniprompt.utils.prompt_utils import make_prompt


def evaluate(data, prompt, config):
    questions, choices, answers = data
    return evaluate_prompt(prompt, questions, answers, choices, config)

def evaluate_prompt(
    new_prompt: str,
    questions: Sequence[str],
    answers: Sequence[str],
    choices: Sequence[List[str]],
    config: Dict[str, Any],
) -> Dict[str, Union[float, List[List[float]]]]:
    acc = 0

    y_true = []
    y_pred = []
    i = 0

    for question, answer, choice in zip(questions, answers, choices):
        i+=1
        if answer not in choice:
            if answer == "1":
                answer = choice[1]
            if answer == "0":
                answer = choice[0]

        prompt = make_prompt(prompt=new_prompt, question=question, choices=choice, template="make_prompt")
        messages = [{"role": "system", "content": "You are an expert"}, {"role": "user", "content": prompt}]
        answer_cot = chat_completion(cache_path=config["cache_path"], **config["solver_llm"], messages=messages)
        answer_llm = extract_answer(answer_cot)

        y_true.append(answer)
        y_pred.append(answer_llm)
        print(f"{i} Answer: {answer}, Predicted: {answer_llm}")

    acc = get_metric(y_true, y_pred, config)
    eval_result =  {
        "acc": acc,
        "cm": get_confusion_matrix(y_true, y_pred, normalize=False).tolist(),
    }
    return eval_result

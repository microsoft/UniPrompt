import os
from typing import Dict, List, Sequence

import pkg_resources
import yaml


def make_prompt(prompt: str, question: str, choices: Sequence[List[str]], template="make_prompt") -> str:
    prompts = load_prompts()
    prompt_template = prompts.get(template, None)
    formatted_prompt = prompt_template.format(
        prompt=prompt,
        question=question,
        choices=choices,
    )
    return formatted_prompt

def make_prompt_code(prompt: str, question: str, choices: Sequence[List[str]]) -> str:
    prompts = load_prompts()
    prompt_template = prompts.get("make_prompt_code", None)
    formatted_prompt = prompt_template.format(
        prompt=prompt,
        question=question,
        choices=choices,
    )
    return formatted_prompt

def load_prompts() -> Dict[str, str]:
    # update the path here for custom.yaml
    prompt_path = pkg_resources.resource_filename("uniprompt", os.path.join("metaprompts", "custom.yaml"))
    # prompt_path = os.path.join(os.path.dirname(__file__), "metaprompts", "custom.yaml")
    with open(prompt_path) as f:
        prompts = yaml.safe_load(f)
    return prompts

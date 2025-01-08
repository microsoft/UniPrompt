import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

from uniprompt.evaluate import evaluate_prompt
from uniprompt.feedback.feedback import apply_edits
from uniprompt.utils.api_utils import chat_completion
from uniprompt.utils.prompt_utils import load_prompts


class BeamSearch:
    def __init__(self, beam_width: int):
        self.beam_width = beam_width
        self.beam = []

    def initialize_candidates(self, initial_prompt: str, data: Tuple, config: Dict, eval_fn: Optional[Callable] = None) -> List[Tuple[float, str, float]]:
        val_questions, val_choices, val_answers = data
        if eval_fn is None:
            eval_fn = evaluate_prompt
        eval_result = eval_fn(initial_prompt, val_questions, val_answers, val_choices, config)
        self.beam.append((-eval_result["acc"], initial_prompt, 0))

    def get_best_prompt(self) -> str:
        return self.beam[0][1]

    def apply_edits(prompt: str, prompt_template: str, edits: str, config: Dict[str, Any]) -> str:
        input_prompt = prompt_template.format(prompt=prompt, edits=edits)
        messages = [{"role": "user", "content": input_prompt}]
        output = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
        return output

    def apply_edits_to_beam(self, final_feedback: str, val_data: Tuple, config) -> None:
        new_candidates = []
        prompts = load_prompts()
        val_questions, val_choices, val_answers = val_data
        for beam_acc, beam_prompt, _ in self.beam:
            new_prompt = apply_edits(
                prompt=beam_prompt,
                prompt_template=prompts.get("apply_edits", None),
                edits=final_feedback,
                config=config,
            )
            eval_result = evaluate_prompt(new_prompt, questions=val_questions, choices=val_choices, answers=val_answers, config=config)
            acc, cm = eval_result["acc"], eval_result["cm"]
            new_candidates.append((-acc, new_prompt, -(beam_acc-acc)))

        self.beam = heapq.nsmallest(self.beam_width, self.beam + new_candidates)
        print(f"New Beam: {self.beam}")
        return self.beam

    def add_prompt_to_beam(self, prompt: str, val_data: Tuple, config: Dict, eval_fn: Optional[Callable] = None) -> None:
        val_questions, val_choices, val_answers = val_data
        if eval_fn is None:
            eval_fn = evaluate_prompt

        eval_result = eval_fn(prompt, questions=val_questions, choices=val_choices, answers=val_answers, config = config)
        new_candidate = (-eval_result["acc"], prompt, (self.beam[0][0] + (eval_result["acc"])))
        self.beam = heapq.nsmallest(self.beam_width, self.beam + [new_candidate])
        print(f"Updated Beam: {self.beam}")

from uniprompt.utils.prompt_utils import load_prompts
from uniprompt.utils.api_utils import chat_completion
import random
import optuna
import heapq

"""
TIPS taken from dspy implementation: https://github.com/stanfordnlp/dspy/blob/540805e1feac687f5ee80c62402cf5d7b3373b29/dspy/propose/grounded_proposer.py#L13
@inproceedings{khattab2024dspy,
  title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines},
  author={Khattab, Omar and Singhvi, Arnav and Maheshwari, Paridhi and Zhang, Zhiyuan and Santhanam, Keshav and Vardhamanan, Sri and Haq, Saiful and Sharma, Ashutosh and Joshi, Thomas T. and Moazam, Hanna and Miller, Heather and Zaharia, Matei and Potts, Christopher},
  journal={The Twelfth International Conference on Learning Representations},
  year={2024}
}
"""

TIPS = {
        "none": "",
        "creative": "Don't be afraid to be creative when creating the new instruction!",
        "simple": "Keep the instruction clear and concise.",
        "description": "Make sure your instruction is very informative and descriptive.",
        "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
        "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
    }

def add_random_tip_to_prompt(prompt: str) -> str:
    random_tip_key = random.choice(list(TIPS.keys()))
    random_tip = TIPS[random_tip_key]
    
    if random_tip:
        prompt_with_tip = f"{prompt}\n\nTip: {random_tip}"
    else:
        prompt_with_tip = prompt
    
    return prompt_with_tip

def extract_task_type(prompt: str) -> str:
    prompts = load_prompts()
    task_type_prompt = prompts.get("task_type", None)
    task_type_prompt = task_type_prompt.format(prompt = prompt)
    messages = [{"role": "user", "content": task_type_prompt}]
    task_type = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
    return task_type

def create_dataset_summary(trainset, view_data_batch_size) -> str:
    prompts = load_prompts()
    dataset_summary_prompt = prompts.get("dataset_summary", None)

    summaries = []
    for start in range(0, len(trainset), view_data_batch_size):
        end = min(start + view_data_batch_size, len(trainset))
        batch = trainset[start:end]

        batch_str = "\n".join([str(item) for item in batch])

        summary_prompt = dataset_summary_prompt.format(examples=batch_str)
        messages = [{"role": "user", "content": summary_prompt}]
        output = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)
        
        summaries.append(output)

    # Combine all batch summaries into a single string
    observations = " ".join(summaries)
    observation_summary_prompt = prompts.get("observation_summary", None)
    final_summary_prompt = observation_summary_prompt.format(observations=observations)
    messages = [{"role": "user", "content": final_summary_prompt}]
    final_summary = chat_completion(cache_path=config["cache_path"], **config["expert_llm"], messages=messages)

    return final_summary

def COT_prompt(prompt: str) -> str:
    cot_prompt= f"{prompt}\n\n Think step by step."
    
    return cot_prompt

def APE_prompt(prompt: str) -> str:
    ape_prompt= f"{prompt}\n\n Lets work this out in a step by step way to be sure we have the right answer."
    
    return ape_prompt

def optimize_prompt(prompt: str, num_trials: int, seed: int) -> str:

    # Define the objective function
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)

def past_prompts_with_evaluation(train_data, beam, config) -> str:
    prompts = load_prompts()
    initial_prompt = prompts.get("opro_initial", None)
    iteration = config["iterations"]

    num_prompts = max(20, len(beam.beam))
    
    top_prompts = heapq.nsmallest(num_prompts, beam.beam)
    
    formatted_prompts = []
    for score, prompt, _ in top_prompts:
        formatted_prompts.append(f"text:\n{prompt}\n\nscore:\n{-score:.2f}\n")
    
    combined_past_prompts = "\n".join(formatted_prompts)

    middle_prompt = prompts.get("opro_middle", None)

    train_questions, train_answers = train_data

    random_indices = random.sample(range(len(train_questions)), min(3, len(train_questions)))
    random_examples = [
        f"Question: {train_questions[i]}\n Answer: <Answer><INS></Answer> \n Output: {train_answers[i]}"
        for i in random_indices
    ]
    train_data_str = "\n".join(random_examples)

    end_prompt = prompts.get("opro_end", None)
    end_prompt = end_prompt.format(num_prompts=iteration)

    final_prompt = f"{initial_prompt}\n\n{combined_past_prompts}\n\n{middle_prompt}\n\n{train_data_str}\n\n{end_prompt}"
    
    return final_prompt

    

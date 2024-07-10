import heapq
import random
import re
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from .logger import setup_logger
from .utils import (
    apply_edits,
    chat_completion,
    combine_multiple_feedbacks_with_examples,
    feedback_one_example,
    feedback_with_history,
    get_confusion_matrix,
    load_dataset,
    load_prompts,
)

MetricType = Literal["accuracy", "weighted_accuracy", "hinge_accuracy"]

class UniPrompt:

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the UniPrompt optimizer.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing settings for the optimizer.
        """
        self.config = config
        logging_file_path = config.get("logging_file_path")
        self.logger = setup_logger(log_file=logging_file_path)
        self.prompts = load_prompts()

    def make_prompt(self, prompt: str, question: str, choices: Sequence[List[str]]) -> str:
        """
        Create a formatted prompt for a given question and choices.

        Args:
            prompt (str): The base prompt.
            question (str): The question to be answered.
            choices (Sequence[List[str]]): The answer choices, if any.

        Returns:
            str: The formatted prompt including the question and answer format instructions.
        """
        prompt_template = self.prompts.get("make_prompt", None)
        formatted_prompt = prompt_template.format(
            prompt=prompt,
            question=question,
            choices=choices,
        )
        return formatted_prompt

    def extract_answer(self, cot: str) -> Optional[str]:
        """
        Extract the answer from the chain of thought response.

        Args:
            cot (str): The chain of thought string containing the answer.

        Returns:
            Optional[str]: The extracted answer, or None if no answer is found.
        """
        pattern = r"<Answer>(.*?)<\/Answer>"
        match = re.search(pattern, cot)
        answer = match.group(1) if match else None
        return answer

    def evaluate_prompt(
        self,
        new_prompt: str,
        questions: Sequence[str],
        answers: Sequence[str],
        choices: Sequence[List[str]],
        config: Dict[str, Any]
    ) -> Dict[str, Union[float, List[List[float]]]]:
        """
        Evaluate the prompt on a set of questions, choices, and answers.

        Args:
            new_prompt (str): The prompt to evaluate.
            questions (Sequence[str]): The list of questions.
            answers (Sequence[str]): The list of correct answers.
            choices (Sequence[List[str]]): The list of answer choices for each question.
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            Dict[str, Union[float, List[List[float]]]]: A dictionary containing the accuracy and confusion matrix.
        """
        acc = 0

        y_true = []
        y_pred = []

        for question, answer, choice in zip(questions, answers, choices):
            prompt = self.make_prompt(prompt=new_prompt, question=question, choices=choice)
            messages = [{"role": "user", "content": prompt}]
            answer_cot = chat_completion(cache_path=config["cache_path"], **config["solver_llm"], messages=messages)
            answer_llm = self.extract_answer(answer_cot)

            y_true.append(answer)
            y_pred.append(answer_llm)

        acc = self.get_metric(y_true, y_pred, config)
        eval_result =  {
            "acc": acc,
            "cm": get_confusion_matrix(y_true, y_pred, normalize=False).tolist(),
        }

        return eval_result

    def get_metric(self, y_true: List[str], y_pred: List[str], config: Dict[str, Any]) -> float:
        """
        Calculate the metric based on true and predicted answers.

        Args:
            y_true (List[str]): List of true answers.
            y_pred (List[str]): List of predicted answers.
            config (Dict[str, Any]): Configuration dictionary containing metric settings.

        Returns:
            float: The calculated metric value.

        Raises:
            ValueError: If an unsupported metric type is specified.
        """
        cm = get_confusion_matrix(y_true, y_pred, normalize=True)
        metric_type: MetricType = config["metric_kwargs"]["type"]

        if metric_type == "accuracy":
            diagonal = cm.diagonal()
            metric_val =  sum(diagonal) / diagonal.shape[0]

        elif metric_type == "weighted_accuracy":
            weights = config["metric_kwargs"]["weights"]
            diagonal = cm.diagonal()
            metric_val = sum(diagonal * weights) / sum(weights)

        elif metric_type == "hinge_accuracy":
            thresholds = config["metric_kwargs"]["thresholds"]
            weights = config["metric_kwargs"]["weights"]
            diagonal = cm.diagonal()
            diagonal.setflags(write=1)
            for i in range(diagonal.shape[0]):
                if diagonal[i] < thresholds[i]:
                    diagonal[i] = diagonal[i] / thresholds[i] * diagonal[i]
            metric_val = sum(diagonal * weights) / sum(weights)

        else:
            raise ValueError(f"Metric {metric_type} not supported")

        return metric_val

    def synthesize_sub_prompts(self, initial_prompt: str, num_samples: int) -> List[str]:
        """
        Create multiple sub-prompts from the initial prompt by dynamically assembling sections.

        Args:
            initial_prompt (str): The seed prompt with multiple sections, each represented by 'Section title: <text>\n\n'.
            num_samples (int): Number of sub-prompts to generate.

        IN this task, you have to answer yes or no

        Meaning of yes: <multiline string>

        Meaning of no: <multiline string>

        1.IN this task, you have to answer yes or no

        2. IN this task, you have to answer yes or no

            Meaning of yes: <multiline string>


        Returns:
            List[str]: List of generated sub-prompts.
        """

        # Split the prompt into sections
        sections = initial_prompt.split("\n\n")

        # Extract the non-sectioned part (first section)
        non_sectioned = sections[0].strip()

        # Process the remaining sections
        section_dict = {}
        for section in sections[1:]:
            # Find the first colon in the section
            colon_index = section.find(":")
            if colon_index != -1:
                # Extract the section name and content
                name = section[:colon_index].strip()
                content = section[colon_index+1:].strip()
                section_dict[name] = content

        sub_prompts = {initial_prompt,} # The initial prompt is part of
        attempts = 0
        max_attempts = num_samples * 10  # Arbitrary multiplier to limit attempts

        while len(sub_prompts) < num_samples and attempts < max_attempts:
            # Randomly select sections
            num_to_select = random.randint(0, len(section_dict))
            selected_sections = random.sample(list(section_dict.items()), num_to_select)

            # Assemble the sub-prompt
            sub_prompt = non_sectioned + "\n\n" + "\n\n".join(f"{name}: {content}" for name, content in selected_sections)

            while sub_prompt[-1] == "\n":
                sub_prompt = sub_prompt[:-1]

            # Add to set if it's a new unique prompt
            sub_prompts.add(sub_prompt)
            attempts += 1

        return list(sub_prompts)

    def make_groups(
        self,
        prompt: str,
        questions: Sequence[str],
        answers: Sequence[str],
        choices: Sequence[List[str]],
        config: Dict[str, Any]
    ) -> Dict[int, List[int]]:
        """
        Group the questions into clusters based on the feedbacks.

        Args:
            prompt (str): The prompt to use when grouping.
            questions (Sequence[str]): The list of questions.
            answers (Sequence[str]): The list of correct answers.
            choices (Sequence[List[str]]): The list of answer choices for each question.
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            Dict[int, List[int]]: A dictionary with cluster numbers as keys and lists of question indices as values.
        """
        self.logger.info("Starting to group questions")
        feedbacks = []
        for question, choice, answer in zip(questions, choices, answers):
            formatted_prompt = self.make_prompt(prompt=prompt, question=question, choices=choice)
            messages = [{"role": "user", "content": formatted_prompt}]
            answer_cot = chat_completion(cache_path=config["cache_path"], **config["solver_llm"], messages=messages)
            answer_llm = self.extract_answer(answer_cot)
            if answer_llm is not None and str(answer_llm) != str(answer):
                wrong_choices = answer_llm
                wrong_cots = answer_cot
                correct_choices = answer
                question_feedback = feedback_one_example(
                    prompt=formatted_prompt,
                    prompt_template=self.prompts.get("feedback_one_example", None),
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

        group_prompt = self.prompts.get("group_prompt", None).format(feedbacks=group_feedbacks)
        messages = [{"role": "user", "content": group_prompt}]
        all_groups = re.findall(r"<Cluster>(.*?)</Cluster>", chat_completion(cache_path=config["cache_path"], **config["grouping_llm"], messages=messages), re.DOTALL)

        groups = {}
        groups[0] = []
        groups_str = "Group 0: Correct Answer\n"
        for i, g in enumerate(all_groups):
            groups[i+1] = [] # i+1 because 0 is reserved for correct answers
            groups_str += f"Group {i}: {g}\n"

        self.logger.debug(f"The groups with their explanations: {repr(groups_str)}")

        for idx, (question, choice, answer, feedback) in enumerate(zip(questions, choices, answers, feedbacks)):
            assign_group_prompt = self.prompts.get("assign_group_prompt", None).format(groups_str=groups_str, feedback=feedback)
            messages = [{"role": "user", "content": assign_group_prompt}]
            cluster_number = chat_completion(cache_path=config["cache_path"], **config["grouping_llm"], messages=messages)
            cluster_number_extracted = int(re.search(r"<Answer>(.*?)</Answer>", cluster_number).group(1))
            groups[cluster_number_extracted].append(idx)

        self.logger.info(f"Grouping completed. Number of groups: {len(groups)}")

        return groups

    def optimize(self) -> str:
        self.logger.info("Starting optimization process...")
        start_time = time.time()

        initial_prompt = self.config["initial_prompt"]
        dataset_path = self.config["dataset_path"]

        # Hyperparameters
        mini_batch_size = self.config["mini_batch_size"]
        batch_size = self.config["batch_size"]
        iterations = self.config["iterations"]
        epochs = self.config["epochs"]
        beam_width = self.config["beam_width"]
        epsilon = self.config["epsilon"]
        group_frequency = self.config["group_frequency"]
        create_sub_prompts = self.config["create_sub_prompts"]
        num_sub_prompts = self.config["num_sub_prompts"]

        dataset_dict = load_dataset(dataset_path)
        self.logger.info(f"Loaded dataset from {dataset_path}")

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

        if create_sub_prompts:
            initial_prompts = self.synthesize_sub_prompts(initial_prompt, num_samples=num_sub_prompts)
        else:
            initial_prompts = [initial_prompt]


        # Initialize beam
        beam = []
        for p in initial_prompts:
            acc = self.evaluate_prompt(p, questions=val_questions, answers=val_answers, choices=val_choices, config=self.config)["acc"]
            beam.append((-acc, p, 0))
        beam = heapq.nsmallest(beam_width, beam)
        best_acc = beam[0][0]

        self.logger.info(f"Initial best accuracy: {-best_acc:.4f}")
        # Re-compute improvements
        for i, (acc, prompt, _) in enumerate(beam):
            beam[i] = (acc, prompt, -(acc - best_acc))
            self.logger.debug(f"Beam {i+1}: Accuracy={-acc:.4f}, Improvement = {beam[i][2]:.4f}, Prompt={repr(prompt)}")


        for epoch in range(epochs):
            _, best_prompt, _ = beam[0]

            # Group questions based on initial prompt
            if epoch % group_frequency == 0:
                groups = self.make_groups(prompt=best_prompt, questions=train_questions, answers=train_answers, choices=train_choices, config=self.config)

            self.logger.info(f"Starting epoch {epoch + 1}/{self.config['epochs']}")

            # Initialize edit history for this prompt
            edit_history_dict = {group: [] for group in groups}

            for _ in range(iterations):
                for group in groups:
                    # Get the best performing prompt
                    _, curr_prompt, _ = beam[0]

                    new_candidates: List[Tuple[float, str]] = []

                    acc_batch = 0
                    feedback = ""
                    try:
                        selected_indices = random.sample(groups[group], k=batch_size*mini_batch_size)
                    except ValueError: # If not enough samples in the group
                        selected_indices = groups[group]

                    batch_questions = [train_questions[index] for index in selected_indices]
                    batch_choices = [train_choices[index] for index in selected_indices]
                    batch_answers = [train_answers[index] for index in selected_indices]

                    for mini_batch_start in range(0, len(batch_questions), mini_batch_size): # Mini-batch loop
                        mini_batch_questions = batch_questions[mini_batch_start:mini_batch_start + mini_batch_size]
                        mini_batch_choices = batch_choices[mini_batch_start:mini_batch_start + mini_batch_size]
                        mini_batch_answers = batch_answers[mini_batch_start:mini_batch_start + mini_batch_size]

                        wrong_questions, wrong_choices, wrong_cots, correct_choices = [], [], [], []

                        # Identifying failure cases of the current prompt
                        for question, answer, choices in zip(mini_batch_questions, mini_batch_answers, mini_batch_choices):
                            prompt = self.make_prompt(prompt=curr_prompt, question=question, choices=choices)
                            messages = [{"role": "user", "content": prompt}]
                            answer_cot = chat_completion(cache_path=self.config["cache_path"], **self.config["solver_llm"], messages=messages)
                            answer_llm = self.extract_answer(answer_cot)

                            if str(answer_llm) != str(answer):
                                wrong_choices.append(answer_llm)
                                wrong_cots.append(answer_cot)
                                wrong_questions.append(question)
                                correct_choices.append(answer)
                            else:
                                acc_batch += 1 / len(batch_questions)

                        if wrong_questions:
                            feedback_new = feedback_with_history(
                                prompt=curr_prompt,
                                prompt_template=self.prompts.get("feedback_with_history", None),
                                questions=wrong_questions,
                                answers=correct_choices,
                                pred_answers=wrong_choices,
                                cots=wrong_cots,
                                history=edit_history_dict[group],
                                config=self.config,
                            )
                            feedback += feedback_new + "====================="

                    self.logger.debug(f"Feedback for group {group}={feedback}")
                    if feedback:
                        forced_feedback = random.uniform(0, 1) # Randomly decide to either add, delete or set a section
                        if forced_feedback < epsilon:
                            section_action = random.choice(("add", "delete", "set")) # Choose one of add, delete or set to perform
                            self.logger.debug(f"Forced_feedback={forced_feedback}, Epsilon={epsilon}, Forcing action {section_action}")
                            final_feedback = combine_multiple_feedbacks_with_examples(
                                prompt_template=self.prompts.get(f"combine_multiple_feedbacks_with_examples_{section_action}", None),
                                edits=feedback,
                                wrong_examples=wrong_questions,
                                config=self.config,
                            )
                        else:
                            final_feedback = combine_multiple_feedbacks_with_examples(
                                prompt_template=self.prompts.get("combine_multiple_feedbacks_with_examples", None),
                                edits=feedback,
                                wrong_examples=wrong_questions,
                                config=self.config,
                            )

                        # Apply edits to all prompts in the beam
                        for beam_acc, beam_prompt, _ in beam:
                            new_prompt = apply_edits(
                                prompt=beam_prompt,
                                prompt_template=self.prompts.get("apply_edits", None),
                                edits=final_feedback,
                                config=self.config,
                            )
                            eval_result = self.evaluate_prompt(new_prompt, questions=val_questions, choices=val_choices, answers=val_answers, config=self.config)
                            acc, cm = eval_result["acc"], eval_result["cm"]
                            new_candidates.append((-acc, new_prompt, -(beam_acc-acc)))

                            self.logger.debug(f"prompt group {group}, accuracy={acc:.4f}, cm={cm}, prompt={repr(new_prompt)}")

                        # Merge new candidates with the current beam and select top-k
                        beam = heapq.nsmallest(beam_width, beam + new_candidates)

                        # Update edit history
                        edit_history_dict[group].append([final_feedback, beam[0][2]])

                    # Log all prompts in the beam
                    for i, (acc, prompt, improvement) in enumerate(beam):
                        train_result = self.evaluate_prompt(prompt, questions=train_questions, choices=train_choices, answers=train_answers, config=self.config)
                        val_result = self.evaluate_prompt(prompt, questions=val_questions, choices=val_choices, answers=val_answers, config=self.config)
                        test_result = self.evaluate_prompt(prompt, questions=test_questions, choices=test_choices, answers=test_answers, config=self.config)
                        self.logger.debug(f"Beam {i+1}: Accuracy={-acc:.4f}, Improvement = {improvement:.4f}, Train Accuracy={train_result['acc']:.4f} {train_result['cm']}, Val Accuracy={val_result['acc']:.4f} {val_result['cm']}, Test Accuracy={test_result['acc']:.4f} {test_result['cm']}, Prompt={repr(prompt)}")

        end_time = time.time()
        self.logger.info(f"Optimization completed in {end_time - start_time:.2f} seconds")

        best_prompt = beam[0][1]
        self.logger.info(f"Final best prompt: {best_prompt}")

        # Final evaluation on test set
        test_acc = self.evaluate_prompt(best_prompt, questions=test_questions, answers=test_answers, choices=test_choices, config=self.config)
        self.logger.info(f"Final test accuracy: {test_acc['acc']:.4f}")

        return best_prompt

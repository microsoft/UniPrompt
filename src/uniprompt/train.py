import random

from uniprompt.feedback.feedback import combine_multiple_feedbacks_with_examples, feedback_with_history
from uniprompt.history.history import process_history
from uniprompt.utils.api_utils import chat_completion, extract_answer
from uniprompt.utils.prompt_utils import load_prompts, make_prompt
from uniprompt.utils.sampling_utils import mini_batch_indices, random_batch


def train(train_data, val_data, config, beam, grouping):
    prompts = load_prompts()
    groups = grouping.groups
    train_questions, train_choices, train_answers = train_data
    for group in groups:
        if(len(groups[group]) == 0):
            continue
        # Get the best performing prompt
        curr_prompt = beam.get_best_prompt()

        batch_acc = 0
        if "mini_batch_size" in config:
            mini_batch_size = config["mini_batch_size"]
        else:
            mini_batch_size = 1

        batch_size = config["batch_size"] * mini_batch_size
        selected_indices = random_batch(groups[group], batch_size)
        batch_questions = [train_questions[index] for index in selected_indices]
        batch_choices = [train_choices[index] for index in selected_indices]
        batch_answers = [train_answers[index] for index in selected_indices]

        global_feedback = ""
        final_feedback = ""
        feedback = []
        correct_answers = []
        total_wrong_questions = []
        if "mini_batch_size" in config:
            mini_batch_ranges = mini_batch_indices(len(batch_questions), mini_batch_size)
            for start, end in mini_batch_ranges: # Mini-batch loop
                mini_batch_questions = batch_questions[start:end]
                mini_batch_choices = batch_choices[start:end]
                mini_batch_answers = batch_answers[start:end]
                # Use a list for the feedback to be able to append to it since strings are immutable and we need to append for each mini batch
                feedback = train_batch_adaptive(mini_batch_questions, mini_batch_choices, mini_batch_answers, curr_prompt, feedback, correct_answers, group, total_wrong_questions, config, grouping)
            global_feedback = " ".join(feedback)

            # Adding randomization to the feedback to force the model to explore more options
            forced_feedback = random.uniform(0, 1) # Randomly decide to either add, delete or set a section
            if forced_feedback < config["epsilon"]:
                section_action = random.choice(("add", "delete", "set")) # Choose one of add, delete or set to perform
                # can send the wrong questions for multiple feedbacks with examples
                final_feedback = combine_multiple_feedbacks_with_examples(
                    prompt_template=prompts.get(f"combine_multiple_feedbacks_with_examples_{section_action}", None),
                    edits=global_feedback,
                    wrong_examples=total_wrong_questions,
                    config=config,)

        else:
            global_feedback = train_batch_adaptive(batch_questions, batch_choices, batch_answers, curr_prompt, global_feedback, correct_answers, group, total_wrong_questions, config, grouping)
            final_feedback = " ".join(global_feedback)

        # apply edit to all prompts in the beam
        beam.beam = beam.apply_edits_to_beam(final_feedback, val_data = val_data, config = config)

    return beam

def train_batch_adaptive(mini_batch_questions, mini_batch_choices, mini_batch_answers, p, feedback, correct_answers, group, total_wrong_questions, config, grouping):
    prompts = load_prompts()
    wrong_questions, wrong_choices, wrong_cots, correct_choices = [], [], [], []
    acc = 0
    for question, answer, choices in zip(mini_batch_questions, mini_batch_answers, mini_batch_choices):
        prompt = make_prompt(prompt=p, question=question, choices=choices)
        messages = [{"role": "user", "content": prompt}]
        answer_cot = chat_completion(cache_path=config["cache_path"], **config["solver_llm"], messages=messages)
        answer_llm = extract_answer(answer_cot)

        if str(answer_llm) != str(answer):
            wrong_choices.append(answer_llm)
            wrong_cots.append(answer_cot)
            wrong_questions.append(question)
            total_wrong_questions.append(question)
            correct_choices.append(answer)
        else:
            acc += 1

    if wrong_questions:
        history = grouping.edit_history_dict[group]
        final_history = process_history(history)
        feedback_new = feedback_with_history(
            prompt=p,
            prompt_template=prompts.get("feedback_with_history", None),
            questions=wrong_questions,
            answers=correct_choices,
            pred_answers=wrong_choices,
            cots=wrong_cots,
            history=final_history,
            config=config,
        )
        feedback.append(feedback_new + "=====================")
        correct_answers.append(acc)

    print(f"Wrong Questions {wrong_questions}")

    return feedback

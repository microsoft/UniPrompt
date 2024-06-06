## Import necessary libraries
from datasets import load_dataset
import re
from utils import *
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import random
import json
import time
import argparse

## Helper Functions
def extract_answer(cot):
    pattern = r'<Answer>(.*?)<\/Answer>'
    match = re.search(pattern, cot)
    if match:
        # ans = re.sub("[^\d\.]", "", match.group(1))
        return match.group(1)
    return None

def make_prompt(prompt, question, answer, choices):

    if(len(choices)>0):
        question_str = f"Question: {question} \n Options: "
        answer_string = ""
        for i in range(len(choices)):
            question_str += f"{chr(65+i)}:{choices[i]}, "
            answer_string += "<Answer>"+chr(65+i)+"</Answer> or"
        made_prompt = f'''
        {prompt}
        output_format: Give your answer as the correct option between <Answer></Answer> tags like: {answer_string[:-2]}.
        Now, answer the following question:
        {question_str}
        '''
        
    else:
        question_str = f"Question: {question} \n"
        made_prompt = f'''
        {prompt}
        output_format: Give your answer as the correct number between <Answer></Answer> tags like: <Answer>201</Answer> or <Answer>5505</Answer> or <Answer>1</Answer> etc.
        Now,answer the following question:
        {question_str}
        '''
    return made_prompt

def evaluate_prompt(new_prompt, questions, answers, choices, solver_llm="gpt-35-turbo"):
    acc=0
    total = 0
    for question, answer, choice in zip(questions, answers, choices):
        prompt = make_prompt(new_prompt, question, answer, choice)
        answer_cot = LLM(prompt, 0, solver_llm)
        answer_llm = extract_answer(answer_cot)
        if(str(answer) in str(answer_llm) or str(answer_llm) in str(answer)):
            acc += 1/len(questions)
        total+=1
    return acc

def log_iteration(iteration_data, file_path):
    with open(file_path, 'a') as json_file:
        json.dump(iteration_data, json_file)
        json_file.write('\n')

## CLustering Function
def make_groups(questions, choices, answers, initial_prompt, task, groupng_llm="gpt-4", solver_llm="gpt-35-turbo"):
    feedbacks = []
    for question, choice, answer in zip(questions, choices, answers):
        prompt = make_prompt(initial_prompt, question, answer, choice)
        answer_cot = LLM(prompt, 0, solver_llm)
        answer_llm = extract_answer(answer_cot)
        if(answer_llm != None and str(answer_llm) != str(answer) ):
            wrong_choices = choice[ord(answer_llm)-65]
            wrong_cots = answer_cot
            correct_choices = choice[ord(answer)-65]
            question_feedback = feedback_one_example(task, initial_prompt, [question], [correct_choices], [wrong_choices], [wrong_cots])
            feedbacks.append(question_feedback)
        else:
            feedbacks.append("Correct Answer")

    feedback = "\n Feedback: ".join(feedbacks)
    
    prompt = f'''You are given a set of feedbacks, you need to cluster them into five groups based on similarity, and then provide a summary of each group. You can use the following feedbacks to cluster: \n {feedback}

    provide each cluster explnation within the following tags: <Cluster></Cluster>'''
    cluster = LLM(prompt, 0, groupng_llm)
    clusters = re.findall(r'<Cluster>(.*?)</Cluster>', cluster, re.DOTALL)

    groups = {}
    groups[0]=[]
    string_of_clusters = "Group 0: Correct Answer \n"
    i = 1
    for cluster in clusters:
        groups[i] = []
        string_of_clusters += f"Group {i}: {cluster} \n"
        i+=1

    i = 0
    for question, choice, answer, feedback in zip(questions, choices, answers, feedbacks):
        prompt = f'''You are given a feedback and a set of clusters, you need to tell which cluster this feedback belongs to. 
        
        The clusters are: \n {string_of_clusters}

        The feedback is: {feedback}

        give your final answer as the number of the correct cluster between <Answer></Answer> tags like: <Answer>1</Answer>.'''
        cluster_number = LLM(prompt, 0, groupng_llm)
        cluster_number_extracted = re.search(r'<Answer>(.*?)</Answer>', cluster_number)
        groups[int(cluster_number_extracted.group(1))].append(i)
        i+=1
    return groups

## Loading Dataset
def load_dataset(dataset_path):
    # Here I want to read dataset from a jsonl file where the dataset is stored in the following format:
    # split, question, choices, answer
    # I will return the questions, choices and answers as lists

    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]

    train_questions = []
    train_choices = []
    train_answers = []
    dev_questions = []
    dev_choices = []
    dev_answers = []
    test_questions = []
    test_choices = []
    test_answers = []
    
    for question in data:
        if(question['split']=='train'):
            train_questions.append(question['question'])
            train_choices.append(question['choices'])
            train_answers.append(question['answer'])
        elif(question['split']=='validation'):
            dev_questions.append(question['question'])
            dev_choices.append(question['choices'])
            dev_answers.append(question['answer'])
        elif(question['split']=='test'):
            test_questions.append(question['question'])
            test_choices.append(question['choices'])
            test_answers.append(question['answer'])

    return train_questions, train_answers, train_choices, dev_questions, dev_answers, dev_choices, test_questions, test_answers, test_choices

## Optimization Function
def optimize(initial_prompt, task, train_questions, train_choices, train_answers, dev_questions, dev_choices, dev_answers, test_questions, test_choices, test_answers, mini_batch, batch_size, iterations, epochs, logging_file_path, solver_llm="gpt-35-turbo", groupng_llm="gpt-4", expert_llm="gpt-4"):
        
        # Clustering train set based on feedback
        groups = make_groups(train_questions, train_choices, train_answers, initial_prompt, task, groupng_llm, solver_llm)

    # Prompt training begins
    # for k in range(k_fold):

        # Initializing the lists to store accuracies
        accuracies_dev = []
        accuracies_test = []
        accuracies_train = []
        training_step = []
        logging_information = []

        # Initializing the beam
        prompt_0 = initial_prompt
        prompt_1 = initial_prompt  
        prompt_2 = initial_prompt 
        
        # Initial evaluation    
        accuracies_dev.append(evaluate_prompt(initial_prompt, dev_questions, dev_answers, dev_choices, solver_llm))
        accuracies_train.append(evaluate_prompt(initial_prompt, train_questions, train_answers, train_choices, solver_llm))
        accuracies_test.append(evaluate_prompt(initial_prompt, test_questions, test_answers, test_choices, solver_llm))
        training_step.append(0)
        print(f'train: {accuracies_train[-1]}, dev: {accuracies_dev[-1]}, test: {accuracies_test[-1]}')
        logging_information={"K-fold": 0, "epoch": '-1', "group": '-1', "accuracies_test": accuracies_test[-1], "accuracies_dev": accuracies_dev[-1], "accuracies_train": accuracies_train[-1], "training_step": training_step[-1], "prompt": prompt_1}
        log_iteration(logging_information, logging_file_path)  
        
        for xx in range(epochs):

            # Initializing edits history
            edit_history_dict = {}
            for group in groups:
                edit_history_dict[group] = []

            # Can go over the dataset for multiple iterations
            for it in range(iterations):
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
                        mini_batch_questions = batch_questions[mini_batch*z_ : mini_batch*(z_+1)]
                        mini_batch_answers = batch_answers[mini_batch*z_ : mini_batch*(z_+1)]
                        mini_batch_choices = batch_choices[mini_batch*z_ : mini_batch*(z_+1)]
                        wrong_questions = []
                        wrong_choices = []
                        wrong_cots = []
                        correct_choices = []

                        # Identifying failure cases of the current prompt
                        for question, answer, choices in zip(mini_batch_questions, mini_batch_answers, mini_batch_choices):
                                done = 0
                                while done ==0:
                                    try:
                                        prompt = make_prompt(prompt_1, question, answer, choices)
                                        answer_cot = LLM(prompt, 0, solver_llm)
                                        answer_llm = extract_answer(answer_cot)
                                        if(str(answer_llm) == str(answer)):
                                            acc_batch += 1/len(batch_questions)
                                        else:                                        
                                            wrong_choices.append(choices[ord(answer_llm)-65])
                                            wrong_cots.append(answer_cot)
                                            wrong_questions.append(question)
                                            correct_choices.append(choices[ord(answer)-65])
                                        total += 1
                                        done = 1
                                    except:
                                        continue
                        
                        # Providing feedback for the failure cases
                        if(len(wrong_questions)>0):
                            feedback_new = feedback_with_history(task, prompt_1, wrong_questions, correct_choices, wrong_choices, wrong_cots, edit_history_dict[group], expert_llm)
                            feedback += feedback_new + "====================="         

                    # Combining feedbacks over mini-batches
                    if(feedback!=""):
                        final_feedback = combine_multiple_feedbacks_with_examples(feedback, wrong_questions, expert_llm)
                        
                        # Applying edits to the beam
                        prompt_2 = apply_edits(prompt_1, final_feedback, feedback, expert_llm)
                        prompt_3 = apply_edits(prompt_0, final_feedback, feedback, expert_llm)

                        # Evaluating the new prompts
                        acc_0 = evaluate_prompt(prompt_0, batch_questions, batch_answers, batch_choices, solver_llm)
                        acc_1 = evaluate_prompt(prompt_1, batch_questions, batch_answers, batch_choices, solver_llm)
                        acc_2 = evaluate_prompt(prompt_2, batch_questions, batch_answers, batch_choices, solver_llm)
                        acc_3 = evaluate_prompt(prompt_3, batch_questions, batch_answers, batch_choices, solver_llm)

                        # Selecting the best prompts
                        text_number_pairs = list(zip([prompt_0, prompt_1, prompt_2, prompt_3], [acc_0, acc_1, acc_2, acc_3]))
                        sorted_pairs = sorted(text_number_pairs, key=lambda x: x[1], reverse=True)
                        top_pair1, top_pair2 = sorted_pairs[:2]
                        prompt_1, prompt_0 = top_pair1[0], top_pair2[0]
                        acc_top, acc_sec_top = top_pair1[1], top_pair2[1]

                        # Evaluating the best prompt
                        accuracies_dev.append(evaluate_prompt(prompt_1, dev_questions, dev_answers, dev_choices, solver_llm))
                        accuracies_test.append(evaluate_prompt(prompt_1, test_questions, test_answers, test_choices, solver_llm))
                        accuracies_train.append(evaluate_prompt(prompt_1, train_questions, train_answers, train_choices, solver_llm))
                        training_step.append(training_step[-1]+1)

                        # Logging the information
                        logging_information={"K-fold": 0, "epoch": xx, "group": group, "accuracies_test": accuracies_test[-1], "accuracies_dev": accuracies_dev[-1], "accuracies_train": accuracies_train[-1], "training_step": training_step[-1], "prompt": prompt_1}
                        log_iteration(logging_information, logging_file_path)  

                        # Updating edit history
                        edit_history_dict[group].append([final_feedback, acc_top-acc_sec_top])
        return prompt_1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read configuration from a JSON file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the configuration
    config_file = args.config

    ## Loading the configuration file
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    ## Hyperparameters
    mini_batch = config['mini_batch']
    batch_size = config['batch_size']
    iterations = config['iterations']
    epochs = config['epochs']
    solver_llm = config['solver_llm']   
    grouping_llm = config['grouping_llm']
    expert_llm = config['expert_llm']

    ## Some initializations
    logging_file_path = config['logging_file_path']
    initial_prompt = config['initial_prompt']
    task = config['task']
    dataset_path = config['dataset_path']

    ## Loading the dataset
    train_questions, train_answers, train_choices, dev_questions, dev_answers, dev_choices, test_questions, test_answers, test_choices = load_dataset(dataset_path)

    ## Optimizing the prompt
    final_prompt = optimize(initial_prompt, task, train_questions, train_choices, train_answers, dev_questions, dev_choices, dev_answers, test_questions, test_choices, test_answers, mini_batch, batch_size, iterations, epochs, logging_file_path, solver_llm, grouping_llm, expert_llm)
    print("Optimization is done! The final prompt is: \n\n")
    print(final_prompt)

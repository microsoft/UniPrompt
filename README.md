# UniPrompt Repository

📄 This repository contains code to implement our paper **UniPrompt**.

## 🛠️ Setting Up

1. **Set Environment Variables**

   You need to set the following environment variables before running the code:
   - `API_KEY`
   - `ENDPOINT`
   - `API_VERSION`

   ### Example Commands

   For **Windows Command Prompt**:
   ```cmd
   set API_KEY=your_api_key
   set ENDPOINT=your_endpoint
   set API_VERSION=your_api_version
   ```

   For **Windows PowerShell**:
   ```powershell
   $env:API_KEY = "your_api_key"
   $env:ENDPOINT = "your_endpoint"
   $env:API_VERSION = "your_api_version"
   ```

2. **Update the Config File**

   Modify the `config/dataset_name.json` file as per your use case. The configuration includes the following parameters:
   ```json
   {
       "dataset_path": "dataset/ethos.jsonl",
       "mini_batch": 5,
       "batch_size": 7,
       "iterations": 1,
       "epochs": 10,
       "logging_file_path": "logs/ethos.jsonl",
       "initial_prompt": "introduction: In this task, you are given a question. You have to solve the question.",
       "task": "question_answering"
   }
   ```

3. **Prepare the Dataset**

   The dataset format is very important. Ensure your dataset is a JSONL file with the following format:
   - `split`: (train, test, validation)
   - `question`: Full question that you want to get answered, including any prefix or postfix statements
   - `choices`: If the answer has choices, it should be a list, like `[monkey, zebra, lion, tiger]`
   - `answer`: Either a digit or an option like `A`, `B`, `C`, etc.

   Example:
   ```jsonl
   {"split": "train", "question": "What is the largest land animal?", "choices": ["monkey", "zebra", "lion", "tiger"], "answer": "D"}
   ```

## 🚀 Running the Optimization

To get the final optimized prompt, run the following command:

```bash
python .\optimize_k_fold.py --config config/ethos.json
```

---

Feel free to contribute and raise issues! 🛠️✨

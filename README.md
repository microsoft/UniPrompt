# UniPrompt

<div align="center">

![UniPrompt: Generating Multiple Facets of a Task in the Prompt](assets/banner.png)

</div>

## About
UniPrompt looks at prompt optimization as that of learning multiple facets of a task from a set of training examples. UniPrompt, consists of a generative model to generate initial candidates for each prompt section; and a feedback mechanism that aggregates suggested edits from multiple mini-batches into a conceptual description for the section. In particular, it can generate long, complex prompts that baseline algorithms are unable to generate.

## Installation

### 📋 Prerequisites

- Python 3 (>= 3.8)

1. Install the latest version of `pip` and `setuptools`

    ```bash
    python3 -m pip install --upgrade pip setuptools
    ```

1. To setup the package locally, run

    ```bash
    python3 -m pip install .
    ```

### 🔧 Setting up

1. **Set Environment Variables**

   You need to set the `OPENAI_API_KEY` environment variables before running the code:
   
   ### Example Commands

   For **Windows Command Prompt**:
   ```cmd
   set OPENAI_API_KEY=your_api_key
   ```

   For **Windows PowerShell**:
   ```powershell
   $env:OPENAI_API_KEY = "your_api_key"
   ```

1. **Update the Config File**

   Modify the `config/dataset_name.json` file as per your use case.
   If you are using an internal endpoint, make sure to set `api_type` to `azure`, `api_base` to your endpoint URL and `api_version` in your dataset config file. If you are using an OpenAI endpoint, then just set api_type to `oai`.
   
   The configuration includes the following parameters:
   ```json
   {
      "dataset_path": "dataset/ethos.jsonl",
      "mini_batch": 5,
      "batch_size": 7,
      "iterations": 1,
      "epochs": 10,
      "logging_file_path": "logs/ethos.jsonl",
      "initial_prompt": "introduction: In this task, you are given a question. You have to solve the question.",
      "metric_kwargs": {
        "type": "hinge_accuracy",
        "weights": [0.45, 0.55],
        "thresholds": [0.6, 0.75]
      },
      "solver_llm": {
        "model_kwargs": {
            "model": "gpt-4-turbo",
            "temperature": 0
        },
        "api_kwargs": {
            "api_type": "",
            "api_base": "",
            "api_version": ""
        }
      },
      "expert_llm": {
        "model_kwargs": {
            "model": "gpt-4-turbo",
            "temperature": 0
        },
        "api_kwargs": {
            "api_type": "",
            "api_base": "",
            "api_version": ""
        }
      },
      "grouping_llm": {
        "model_kwargs": {
            "model": "gpt-4-turbo",
            "temperature": 0
        },
        "api_kwargs": {
            "api_type": "",
            "api_base": "",
            "api_version": ""
        }
      }
   }
   ```
   Metric `type` can be one of `['accuracy', 'weighted_accuracy', 'hinge_accuracy']` 
   Example config files can be found at [config/ethos.json](config/ethos.json), [config/gk.json](config/gk.json).
   Make sure to set `api_kwargs` before using them.

1. **Prepare the Dataset**

   The dataset format is very important. Ensure your dataset is a JSONL file with the following format:
   - `split`: (train, test, validation)
   - `question`: Full question that you want to get answered, including any prefix or postfix statements
   - `choices`: If the answer has choices, it should be a list, like `[monkey, zebra, lion, tiger]`
   - `answer`: Either a digit or an option like `A`, `B`, `C`, etc.

   Example:
   ```jsonl
   {"split": "train", "question": "What is the largest land animal?", "choices": ["monkey", "zebra", "lion", "tiger"], "answer": "D"}
   ```

### 🚀 Running the Optimization

To get the final optimized prompt, you need to run the `optimize` function from the library:

```python
from uniprompt import optimize

with open(config_file, "r") as f:
      config = json.load(f)

final_prompt = optimize(config=config)
```

For a working example, run
```bash
python examples/main.py --config=config/gk.json
```
Or
```bash
pip install datasets
python examples/ethos.py --config=config/ethos.json
```

## Contributing

### 🛠️ Setup

```bash
pip install -e "./[dev]"
```

### 🖌️ Style guide

To ensure your code follows the style guidelines, install `ruff ~= 4.0`

```shell
ruff check --fix
```
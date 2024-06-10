# UniPrompt

<div align="center">

![UniPrompt: Generating Multiple Facets of a Task in the Prompt](assets/banner.png)

[License](https://github.com/microsoft/UniPrompt/blob/main/LICENSE) |
[Security](https://github.com/microsoft/UniPrompt/blob/main/SECURITY.md) |
[Support](https://github.com/microsoft/UniPrompt/blob/main/SUPPORT.md) |
[Code of Conduct](https://github.com/microsoft/UniPrompt/blob/main/CODE_OF_CONDUCT.md)

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

    Or

    ```bash
    python3 -m pip install git+https://github.com/microsoft/UniPrompt.git
    ```

### 🔧 Setting up

1. **Set Environment Variables**

   You need to set the `OPENAI_API_KEY` environment variables before running the code:
   
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

1. **Update the Config File**

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
      "solver_llm": {
         "model": "gpt-4",
         "temperature": 0
      },
      "expert_llm": {
         "model": "gpt-4",
         "temperature": 0
      },
      "grouping_llm": {
         "model": "gpt-4",
         "temperature": 0
      }
   }
   ```
   Example config files can be found at [config/ethos.json](config/ethos.json) and [config/gk.json](config/gk.json).

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

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### 🛠️ Setup

```bash
pip install -e "./[dev]"
```

### 🖌️ Style guide

To ensure your code follows the style guidelines, install `black>=23.1` and `isort>=5.11`

```shell
pip install black>=24.1.0
pip install isort>=5.12.0
```

then run,

```shell
isort . --sp=pyproject.toml
black . --config=pyproject.toml
```
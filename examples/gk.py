import argparse
import json

from uniprompt import optimize


def main(args):
    # Load the configuration
    config_file = args.config

    # Loading the configuration file
    with open(config_file, "r") as f:
        config = json.load(f)

    final_prompt = optimize(config=config)
    print(f"Optimization is done! The final prompt is: \n\n{final_prompt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read configuration from a JSON file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    main(args)

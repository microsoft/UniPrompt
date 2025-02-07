import os
from typing import Dict, List, Sequence, Any
from pathlib import Path
import pkg_resources
import yaml
from ruamel.yaml import YAML

def make_prompt(prompt: str, question: str, template="make_prompt") -> str:
    prompts = load_prompts()
    prompt_template = prompts.get(template, None)
    formatted_prompt = prompt_template.format(
        prompt=prompt,
        question=question,
    )
    return formatted_prompt

def make_prompt_code(prompt: str, question: str) -> str:
    prompts = load_prompts()
    prompt_template = prompts.get("make_prompt_code", None)
    formatted_prompt = prompt_template.format(
        prompt=prompt,
        question=question,
    )
    return formatted_prompt

def load_prompts() -> Dict[str, str]:
    # update the path here for custom.yaml
    prompt_path = pkg_resources.resource_filename("uniprompt", os.path.join("metaprompts", "custom.yaml"))
    # prompt_path = os.path.join(os.path.dirname(__file__), "metaprompts", "custom.yaml")
    with open(prompt_path) as f:
        prompts = yaml.safe_load(f)
    return prompts

def get_default_metaprompts_path() -> str:
    return pkg_resources.resource_filename("uniprompt", os.path.join("metaprompts", "custom.yaml"))

def update_config_sections(new_sections: Dict[str, Any], config_path: str = None) -> None:
    """
    Updates specific sections in the YAML file with new content while preserving other sections and formatting.
    
    Args:
        new_sections: Dictionary containing the sections to update with their new content
        config_path: Optional path to the YAML file. If not provided, uses the default custom.yaml
    """
    if config_path is None:
        config_path = get_default_metaprompts_path()

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.explicit_end = False
    
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        output_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            for key, new_value in new_sections.items():
                if line.strip().startswith(f"{key}:"):
                    output_lines.append(line)
                    i += 1
                    while i < len(lines) and (lines[i].startswith(' ') or not lines[i].strip()):
                        i += 1
                    new_content = new_value.rstrip().split('\n')
                    for content_line in new_content:
                        output_lines.append(f"  {content_line}\n")
                    if i < len(lines):
                        output_lines.append('\n')
                    break
            else:
                output_lines.append(line)
                i += 1

        with open(config_path, 'w') as f:
            f.writelines(output_lines)
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    except Exception as e:
        raise Exception(f"Error updating YAML configuration file: {e}")
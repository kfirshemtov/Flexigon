import json
import subprocess
import argparse
import os
from ruamel.yaml import YAML


def main(json_path, yaml_path, script_path):
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve relative paths based on the script directory
    json_path = os.path.join(script_dir, json_path) if not os.path.isabs(json_path) else json_path
    yaml_path = os.path.join(script_dir, yaml_path) if not os.path.isabs(yaml_path) else yaml_path
    script_path = os.path.join(script_dir, script_path) if not os.path.isabs(script_path) else script_path

    # Load the prompt list
    with open(json_path, "r") as f:
        prompts = json.load(f)

    # Load YAML config
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(yaml_path, "r") as f:
        config = yaml.load(f)

    # Set default output folder name based on JSON file name
    config['output_folder_name'] = os.path.splitext(os.path.basename(json_path))[0]

    for i, prompt_val in enumerate(prompts):
        full_prompt = f"{args.prefix} {prompt_val}"

        # Reload YAML each iteration to avoid accumulating changes
        with open(yaml_path, "r") as f:
            config = yaml.load(f)

        config['output_folder_name'] = os.path.splitext(os.path.basename(json_path))[0]
        config["prompt"] = full_prompt
        config["override_prev_decimate"] = i == 0
        config["mesh_name"] = ' '.join(prompt_val.split()[-2:]).replace(' ','_') # set the name to the last 2 words

        # Save updated YAML in temporary folder
        updated_cfg_path = '/tmp/cfg.yaml'
        with open('/tmp/cfg.yaml', "w") as f:
            yaml.dump(config, f)

        # Print progress
        print(f"[{i + 1}/{len(prompts)}] Running for prompt: {prompt_val}")

        # Run the script
        subprocess.run(["python3", script_path,'--config_path',updated_cfg_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D shapekeys script for a list of prompts.")
    parser.add_argument("--json_path", type=str, default="prompts_dict/test.json",help="Path to the JSON file with prompts")
    parser.add_argument("--yaml_path", type=str, default="configs/config_multires_high_scale.yaml", help="Path to the YAML config")
    parser.add_argument("--script_path", type=str, default="magic_shapekeys_multires.py",help="Path to the script to run")
    parser.add_argument("--prefix", type=str, default="3D model photorealistic portrait of ",help="Prefix to start each prompt")

    args = parser.parse_args()
    main(args.json_path, args.yaml_path, args.script_path)

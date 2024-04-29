import argparse
from omegaconf import OmegaConf
import yaml
import os
from pathlib import Path
import subprocess

def run_command(command):
    try:
        # Run the command and capture its output
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # Check if the command executed successfully
        if result.returncode == 0:
            # If successful, return the output
            return result.stdout
        else:
            # If there was an error, print the error message
            print("Error:", result.stderr)
            return None
    except FileNotFoundError:
        print("No file found")
        return None
    
def get_filenames(directory_path):
    filenames = []
    # List all files in the directory
    with os.scandir(directory_path) as entries:
        for entry in entries:
            # Check if it's a file (not a directory)
            if entry.is_file():
                if 'data' in entry.name:
                    continue
                else:
                    filenames.append(directory_path + '/' + entry.name)
            else:
                filenames += get_filenames(directory_path + '/' + entry.name)
    return filenames

def load_config_file(config_file_path):
    try:
        # Load configuration from YAML file
        with open(config_file_path, 'r') as file:
            config = OmegaConf.load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None

def modify_config(config, key, value):
    try:
        OmegaConf.update(config, key, value)
        return config
    except Exception as e:
        print(f"Error modifying configuration: {e}")
        return None

def save_config_file(config, config_file_path):
    try:
        with open(config_file_path, 'w') as file:
            OmegaConf.save(config, file)
    except Exception as e:
        print(f"Error saving configuration file: {e}")


def save_no_wv_configs(config, config_save_path=None):
    n_concept = config.data_args.n_concept
    modified_config = modify_config(config, "model_args.no_wv", True)
    modified_config = modify_config(config, "save_dir", "./no_wv_" + str(n_concept))
    modified_config = modify_config(config, "optim_args.learning_rate", 0.01)

    save_config_file(modified_config, "no_wv/" + "config_n" + str(n_concept) +".yaml")




if __name__ == "__main__":

    allfilenames = get_filenames("config")
    # root_dir = '/net/scratch/yiboj/mem-llm'

    root_dir = "./"

    for filname in allfilenames:
        if "yaml" in filname:
            config = load_config_file(filname)

            save_dir = Path(root_dir) / Path(config.save_dir)

            try:
                cmd = "python main.py eval=True eval_path=" + str(save_dir)

                result = run_command(cmd)
                print(str(save_dir))
                print(result)

            except FileNotFoundError:
                print("File not Found")




import argparse
from omegaconf import OmegaConf
import yaml
import os
from pathlib import Path
import subprocess

def delete_files_in_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over the files in the folder
        for file_name in os.listdir(folder_path):
            # Construct the full path to the file
            file_path = os.path.join(folder_path, file_name)
            try:
                # Attempt to remove the file
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    print(f"Skipped: {file_path} (not a file)")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"The folder '{folder_path}' does not exist.")

    
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


def save_no_wv_configs(config):
    n_concept = config.data_args.n_concept
    modified_config = modify_config(config, "model_args.no_wv", True)
    modified_config = modify_config(config, "save_dir", "./no_wv_" + str(n_concept))
    modified_config = modify_config(config, "optim_args.learning_rate", 0.01)

    save_config_file(modified_config, "no_wv/" + "config_n" + str(n_concept) +".yaml")


def save_dim_freeze_configs(config):
    n_concept = config.data_args.n_concept
    if n_concept >= 9:
        return 
    modified_config = modify_config(config, "model_args.freeze_embed", True)
    modified_config = modify_config(config, "root_dir", "/net/scratch/yiboj/mem-llm/dim")

    for dim in [16, 32, 64, 128, 256, 512]:
        modified_config = modify_config(config, "save_dir", "./dim_freeze_n" + str(n_concept)+ "_d" + str(dim) )
        modified_config = modify_config(config, "model_args.dim", dim)
        if dim <=32:
            modified_config = modify_config(config, "optim_args.learning_rate", 0.01)
        else:
            modified_config = modify_config(config, "optim_args.learning_rate", 0.001)

        save_config_file(modified_config, "dim_freeze/" + "config_n" + str(n_concept) + "_d" + str(dim) +".yaml")

def save_dim_no_freeze_configs(config):
    n_concept = config.data_args.n_concept
    if n_concept >= 9:
        return 
    modified_config = modify_config(config, "model_args.freeze_embed", False)
    modified_config = modify_config(config, "root_dir", "/net/scratch/yiboj/mem-llm/dim")

    for dim in [16, 32, 64, 128, 256, 512]:
        modified_config = modify_config(config, "save_dir", "./dim_no_freeze_n" + str(n_concept)+ "_d" + str(dim) )
        modified_config = modify_config(config, "model_args.dim", dim)
        if dim <=32:
            modified_config = modify_config(config, "optim_args.learning_rate", 0.01)
        else:
            modified_config = modify_config(config, "optim_args.learning_rate", 0.001)

        save_config_file(modified_config, "dim_no_freeze/" + "config_n" + str(n_concept) + "_d" + str(dim) +".yaml")

if __name__ == "__main__":
    import shutil

    allfilenames = get_filenames("standard")

    if not os.path.exists("no_wv"):
        os.makedirs("no_wv")
    else:
        delete_files_in_folder("no_wv")

    if not os.path.exists("dim_freeze"):
        os.rmdir("dim_freeze")
    else:
        delete_files_in_folder("dim_freeze")

    if not os.path.exists("dim_no_freeze"):
        os.rmdir("dim_no_freeze")
    else:
        delete_files_in_folder("dim_no_freeze")

    for filname in allfilenames:
        
        config = load_config_file(filname)
        """ save no_wv configs"""
            
        save_no_wv_configs(config)

        config = load_config_file(filname)

        save_dim_freeze_configs(config)

        config = load_config_file(filname)
        save_dim_no_freeze_configs(config)


    # Load configuration from the specified YAML file
    # config = load_config_file(args.config_file_path)

    # if config:
    #     # Modify the configuration
    #     modified_config = modify_config(config, args.key, args.value)

    #     if modified_config:
    #         # Save the modified configuration back to the same file
    #         save_config_file(modified_config, args.config_file_path)
    #     else:
    #         print("Failed to modify configuration. Exiting...")
    # else:
    #     print("Configuration file not loaded. Exiting...")

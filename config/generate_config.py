import argparse
from omegaconf import OmegaConf
import yaml
import os


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

    allfilenames = get_filenames("standard")

    for filname in allfilenames:
        
        config = load_config_file(filname)
        """ save no_wv configs"""
        if not os.path.exists("no_wv"):
            os.makedirs("no_wv")
        save_no_wv_configs(config)
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
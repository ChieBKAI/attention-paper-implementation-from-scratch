import os
from pathlib import Path

# Get weights file path
def get_weights_file_path(config, epoch):
    return str(Path(config['model_folder']) / (config['model_filename'] + str(epoch) + '.pt'))
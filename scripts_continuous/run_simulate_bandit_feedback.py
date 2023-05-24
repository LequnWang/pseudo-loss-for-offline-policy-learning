from exp_utils import data_generation_commands, submit_commands
from exp_params import *
import os

if __name__ == "__main__":
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    commands = data_generation_commands(exp_dir, data_names, data_sizes, n_runs)
    print(len(commands))
    submit_commands(exp_token+"data", exp_dir, split_size, commands, submit=True, shuffle=True)

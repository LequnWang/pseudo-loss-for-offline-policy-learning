from exp_utils import model_fitting_commands, submit_commands
from exp_params import *
import os

if __name__ == "__main__":
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    commands = model_fitting_commands(exp_dir, data_names, data_sizes, max_data_size_for_EB, estimators, oracles, lrs, weight_decays, betas, ks, hs, n_runs)
    print(len(commands))
    submit_commands(exp_token + "fit", exp_dir, split_size, commands, submit=True, shuffle=True)

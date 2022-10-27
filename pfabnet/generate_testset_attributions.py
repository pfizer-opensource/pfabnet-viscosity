import os
import pandas as pd
import glob
import argparse

import numpy as np
import torch



from model import ViscosityNet
from utils import prepare_test_input
from utils import calculate_attribution_grid
from utils import seed_everything


device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()


def get_cnn_models(args, model_files):
    models = []
    for model_file in model_files:
        model = ViscosityNet(args['grid_dim'])
        if os.path.exists(model_file):
            print('loading %s...' % model_file)
            model.load_state_dict(torch.load(model_file))
            model.eval()

        model = torch.nn.DataParallel(model).to(device)
        models.append(model)

    return models


def calculate_test_set_attribution_scores(args, models):
    df = pd.read_csv(args.test_data_file)

    args = vars(args)
    X, _ = prepare_test_input(df, args)

    attribution_scores = []
    for model in models:
        for i in range(len(X)):
            attribution_grid, esp_grid = calculate_attribution_grid(model, X[i], device)
            attribution_grid = attribution_grid[np.abs(esp_grid) > 1e-5]
            attribution_scores.extend(attribution_grid.flatten())

    return np.array(attribution_scores)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate attributions using PfAbNet models')
    parser.add_argument('--test_data_file', type=str, help='test set csv files with entity names')
    parser.add_argument('--homology_model_dir', type=str, help='homology model directory')
    parser.add_argument('--PfAbNet_model_prefix', type=str, default='PfAbNet', help='PfAbNet model prefix')
    parser.add_argument('--PfAbNet_model_dir', type=str, help='PfAbNet model directory')
    parser.add_argument('--grid_dim', type=int, default=96,
                        help='number of grid points along each axis (default = 96)')
    parser.add_argument('--grid_spacing', type=float, default=0.75,
                        help='spacing between grid points (default = 0.75 Angstrom)')
    parser.add_argument('--shell_width', type=float, default=2.0,
                        help='thickness of the surface shell (default 2.0 Angstrom)')
    parser.add_argument('--NX', type=int, default=1,
                        help='number of rotated structures for each input structure (default 1)')
    parser.add_argument('--processors', type=int, default=1,
                        help='Number of CPUs for ESP grid calculation (default 1)')
    parser.add_argument('--esp_dir', type=str, default='', help='directory with precomputed ESP grids')
    parser.add_argument('--output_attribution_scores', type=str, help='file to save attribution scores')
    parser.add_argument('--output_attribution_threshold', type=str, help='file to save attribution threshold')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()

    seed_everything(42)

    model_files_prefix = os.path.join(args.PfAbNet_model_dir, args.PfAbNet_model_prefix)
    model_files = glob.glob('%s*.pt' % model_files_prefix)
    cnn_models = get_cnn_models(vars(args), model_files)

    attribution_scores = calculate_test_set_attribution_scores(args, cnn_models)
    np.save(args.output_attribution_scores, attribution_scores)
    np.save(args.output_attribution_threshold, np.std(attribution_scores))


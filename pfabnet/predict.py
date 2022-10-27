import pandas as pd
import torch
from dataset import ViscosityDataset
from torch.utils.data.dataloader import DataLoader

import argparse
import glob
import os

import numpy as np

from model import ViscosityNet
from utils import seed_everything
from utils import generate_esp_grids, get_esp_grids
from utils import DEFAULT_GRID_PARAMS, ESP_DIR_KEY
from base import ENTITY_KEY


device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()

def get_cnn_models(args, model_files):
    models = []
    for model_file in model_files:
        model = ViscosityNet(args.grid_dim)
        if os.path.exists(model_file):
            print('loading %s...' % model_file)
            model.load_state_dict(torch.load(model_file))
            model.eval()

        model = model.to(device)
        models.append(model)

    return models


def predict(cnn_models, mol_file, args = DEFAULT_GRID_PARAMS):
    if len(args[ESP_DIR_KEY]) > 0:
        esp_grids = get_esp_grids(args, mol_file)
    else:
        esp_grids = generate_esp_grids(args, mol_file)

    esp_grids = [esp_array for esp_array, _ in esp_grids]
    # esp_grids = generate_esp_grids(args, mol_file)
    dummy_y = [0.0]*len(esp_grids)

    test_dataset = ViscosityDataset(esp_grids, dummy_y)

    loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                        batch_size=1, num_workers=0)

    y_preds = []
    for it, d_it in enumerate(loader):
        x, y = d_it

        # place data on the correct device
        x = x.to(device)

        for model in cnn_models:
            # forward the model
            with torch.set_grad_enabled(False):
                output = model(x)

            y1 = output.detach().cpu().squeeze(1).numpy()
            y_preds.extend(y1)


    return np.power(10, np.mean(np.array(y_preds)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate predictions using PfAbNet models')
    parser.add_argument('--structure_file', type=str, help='Input Fv structure')
    parser.add_argument('--PfAbNet_model_prefix', type=str, default='PfAbNet', help='output model prefix')
    parser.add_argument('--PfAbNet_model_dir', type=str, help='output model directory')
    parser.add_argument('--grid_dim', type=int, default=96,
                        help='number of grid points along each axis (default = 96)')
    parser.add_argument('--grid_spacing', type=float, default=0.75,
                        help='spacing between grid points (default = 0.75 Angstrom)')
    parser.add_argument('--shell_width', type=float, default=2.0,
                        help='thickness of the surface shell (default 2.0 Angstrom)')
    parser.add_argument('--NX', type=int, default=10,
                        help='augmentation level (default 10x)')
    parser.add_argument('--processors', type=int, default=5,
                        help='Number of CPUs for ESP grid calculation (default 5)')
    parser.add_argument('--esp_dir', type=str, default='', help='directory with precomputed ESP grids')
    parser.add_argument('--output_file', type=str, help='Output file with prediction')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()

    seed_everything(42)

    model_files_prefix = os.path.join(args.PfAbNet_model_dir, args.PfAbNet_model_prefix)
    model_files = glob.glob('%s*.pt' % model_files_prefix)
    cnn_models = get_cnn_models(args, model_files)

    output = []
    ypred = predict(cnn_models, args.structure_file, args.__dict__)
    output.append({ENTITY_KEY:os.path.basename(args.structure_file).split('.mol2')[0], 'VISCOSITY_PRED':ypred})
    print(args.structure_file, ypred)

    df = pd.DataFrame(output)
    df.to_csv(args.output_file, index=False)





import os
import pandas as pd
import torch
import torch.nn as nn

import argparse
from sklearn.model_selection import KFold

from dataset import ViscosityDataset
from model import ViscosityNet
from trainer import Trainer, TrainerConfig
from utils import seed_everything, prepare_training_input
from base import VISCOSITY_KEY


def train(args):
    seed_everything(42)

    training_data_files = args.training_data_file.split(',')
    df_list = []
    for training_data_file in training_data_files:
        if training_data_file.endswith('.csv'):
            df = pd.read_csv(training_data_file)
        else:
            df = pd.read_pickle(training_data_file)

        df_list.append(df)

    df = pd.concat(df_list)
    df.loc[df[VISCOSITY_KEY] > 1000, VISCOSITY_KEY] = 1000

    X, y = prepare_training_input(df, args.__dict__)

    kf = KFold(n_splits=10, shuffle=True)
    train_index, val_index = list(kf.split(y))[args.fold_idx]

    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    print('Number of datapoints; train: %d, val: %d' % (len(y_train), len(y_val)))

    train_dataset = ViscosityDataset(X_train, y_train)
    val_dataset = ViscosityDataset(X_val, y_val)

    # save model path
    ckpt_file = '%s_%d.pt' % (args.output_model_prefix, args.fold_idx)
    ckpt_path = os.path.join(args.output_model_dir, ckpt_file)
    print('PyTorch model will be saved in ', ckpt_path)

    def weights_init(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model = ViscosityNet(args.num_channels)
    model.apply(weights_init)
    if os.path.exists(ckpt_path):
        print('loading saved model...')
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

    print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'model parameters')

    bs = 1

    history_file = '%s_hist_%d.pkl' % (args.output_model_prefix, args.fold_idx)
    history_path = os.path.join(args.output_model_dir, history_file)
    tconf = TrainerConfig(max_epochs=2000, batch_size=bs, learning_rate=1e-5,
                          num_workers=0, ckpt_path=ckpt_path, history_path=history_path)

    trainer = Trainer(model, train_dataset, val_dataset, tconf)
    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train PfAbNet model')
    parser.add_argument('--training_data_file', type=str, help='training data file')
    parser.add_argument('--homology_model_dir', type=str, help='homology model directory')
    parser.add_argument('--output_model_prefix', type=str, default='PfAbNet', help='output model prefix')
    parser.add_argument('--output_model_dir', type=str, help='output model directory')
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
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of input channels (2 for eisenberg phobic + philic or 3 for esp_eisenberg (default 2))')
    parser.add_argument('--eisenberg_dir', type=str, default='', help='directory with precomputed Eisenberg grids')
    parser.add_argument('--fold_idx', default=0, type=int,
                        help='index of the k-fold split (default = 0)')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()

    os.makedirs(args.output_model_dir, exist_ok=True)

    train(args)




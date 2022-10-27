import os
import random
import multiprocessing
import pickle

import numpy as np

import torch
from captum.attr import IntegratedGradients

from openeye import oechem
from openeye import oegrid
from openeye import oezap
from openeye import oespicoli

try:
    from base import VISCOSITY_KEY, ENTITY_KEY
except Exception as e:
    from .base import VISCOSITY_KEY, ENTITY_KEY

ESP_GRID_KEY = 'ESP_GRID'

INPUT_MOL_KEY = 'INPUT_MOL'
ROT_X_KEY = 'rot_x'
ROT_Y_KEY = 'rot_y'
ROT_Z_KEY = 'rot_z'
GRID_SPACING_KEY = 'grid_spacing'
GRID_DIM_KEY = 'grid_dim'
SHELL_WIDTH_KEY = 'shell_width'
NX_KEY = 'NX' # augmentation level
PROCESSORS_KEY = 'processors'
HOMOLOGY_MODEL_DIR_KEY = 'homology_model_dir'
ESP_DIR_KEY = 'esp_dir'

DEFAULT_GRID_PARAMS = {GRID_DIM_KEY: 96, GRID_SPACING_KEY: 0.75,
                       SHELL_WIDTH_KEY: 2.0, NX_KEY: 10}

def get_molecule(input_file, perceive_residue=True, center_mol=True):
    ifs = oechem.oemolistream(input_file)
    mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, mol)
    ifs.close()

    if perceive_residue:
        oechem.OEPerceiveResidues(mol)
    if center_mol:
        oechem.OECenter(mol)

    return mol


def get_esp_array(params):
    mol = params[INPUT_MOL_KEY]
    theta_x = params[ROT_X_KEY]
    theta_y = params[ROT_Y_KEY]
    theta_z = params[ROT_Z_KEY]
    grid_spacing = params[GRID_SPACING_KEY]
    grid_dim = params[GRID_DIM_KEY]
    shell_width = params[SHELL_WIDTH_KEY]

    oechem.OEEulerRotate(mol, oechem.OEDoubleArray([theta_x, theta_y, theta_z]))

    oechem.OEAssignBondiVdWRadii(mol)

    zap = oezap.OEZap()
    zap.SetInnerDielectric(2.0)
    zap.SetGridSpacing(grid_spacing)
    zap.SetMolecule(mol)

    grid = oegrid.OEScalarGrid(grid_dim, grid_dim, grid_dim,
                               0.0, 0.0, 0.0, grid_spacing)
    zap.SetOuterDielectric(80)
    zap.CalcPotentialGrid(grid)

    surf = oespicoli.OESurface()
    oespicoli.OEMakeMolecularSurface(surf, mol)

    surf_grid = oegrid.OEScalarGrid(grid_dim, grid_dim, grid_dim, 0.0, 0.0, 0.0, grid_spacing)
    oespicoli.OEMakeGridFromSurface(surf_grid, surf)

    grid_size = grid.GetSize()
    arr = np.zeros(grid_size)
    idx = 0
    count = 0
    for i in range(0, grid_dim):
        for j in range(0, grid_dim):
            for k in range(0, grid_dim):
                v = surf_grid.GetValue(i, j, k)
                if 0 <= v < shell_width:
                    val = grid.GetValue(i, j, k)
                    arr[idx] = val

                    count += 1
                idx += 1

    arr3d_esp = np.reshape(arr, (grid_dim, grid_dim, grid_dim, 1))

    return arr3d_esp, mol



def prepare_cnn_input(df, args, train=True):
    hm_model_dir = args[HOMOLOGY_MODEL_DIR_KEY]
    if hm_model_dir is None:
        raise Exception('Homology model directory not specified')

    X = []
    y = []
    for row_idx, row in df.iterrows():
        entity = row[ENTITY_KEY]

        mol_file = os.path.join(hm_model_dir, entity + '.mol2')
        if len(args[ESP_DIR_KEY]) > 0:
            esp_grids = get_esp_grids(args, mol_file)
        else:
            esp_grids = generate_esp_grids(args, mol_file)

        esp_grids = [esp_array for esp_array, _ in esp_grids]

        X.extend(esp_grids)
        if train:
            log_visc = np.log10(row[VISCOSITY_KEY])
            y.extend([log_visc] * args[NX_KEY])
        else:
            y.extend([0.0] * args[NX_KEY])

    return np.array(X), np.array(y)


def get_esp_grids(args, mol_file):
    esp_dir = args[ESP_DIR_KEY]
    esp_array_output = []
    for i in range(args[NX_KEY]):
        with open('%s/rotation_%d/%s.pyb' % (esp_dir, i + 1,
                                             os.path.basename(mol_file).split('.mol2')[0]), 'rb') as fptr:

            mol = get_molecule(os.path.join(os.path.join(esp_dir, 'rotation_%d' % (i+1)), os.path.basename(mol_file)))
            esp_array_output.append((pickle.load(fptr), mol))

    return esp_array_output


def generate_esp_grids(args, mol_file):
    mol = get_molecule(mol_file)

    params = []
    for i in range(args[NX_KEY]):
        rot_x = np.random.uniform(0, 180)
        rot_y = np.random.uniform(0, 180)
        rot_z = np.random.uniform(0, 180)

        params.append({INPUT_MOL_KEY: oechem.OEGraphMol(mol), ROT_X_KEY: rot_x,
                       ROT_Y_KEY: rot_y, ROT_Z_KEY: rot_z,
                       GRID_DIM_KEY: args[GRID_DIM_KEY],
                       GRID_SPACING_KEY: args[GRID_SPACING_KEY],
                       SHELL_WIDTH_KEY: args[SHELL_WIDTH_KEY]})
    if multiprocessing.cpu_count() >= args[PROCESSORS_KEY]:
        processors = args[PROCESSORS_KEY]
    else:
        processors = multiprocessing.cpu_count()
    p = multiprocessing.Pool(processes=processors)
    esp_array_output = p.map(get_esp_array, params)
    p.close()

    output = [(np.moveaxis(esp_array, 3, 0), output_mol) for esp_array, output_mol in esp_array_output]
    return output


def prepare_training_input(df, args):
    return prepare_cnn_input(df, args, train=True)


def prepare_test_input(df, args):
    return prepare_cnn_input(df, args, train=False)


def calculate_attribution_grid(model, esp_grid_in, device='cpu'):
    esp_grid = torch.Tensor(esp_grid_in)
    baseline = torch.zeros(esp_grid.shape)
    esp_grid = esp_grid.unsqueeze(0)
    esp_grid2 = esp_grid.to(device)

    baseline = torch.unsqueeze(baseline, 0)
    baseline = baseline.to(device)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(esp_grid2, baseline, target=0, return_convergence_delta=True)
    attributions = attributions.detach().cpu().numpy()
    esp_grid2 = esp_grid2.detach().cpu().numpy()

    return attributions, esp_grid2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

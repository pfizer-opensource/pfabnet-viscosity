import os
import random
import multiprocessing
import pickle
import collections

import numpy as np

import torch
from openeye import oechem
from openeye import oegrid
from openeye import oezap
from openeye import oespicoli

try:
    from base import VISCOSITY_KEY, ENTITY_KEY
except Exception as e:
    from .base import VISCOSITY_KEY, ENTITY_KEY

EISENBERG_GRID_KEY = 'EISENBERG_GRID'

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
EISENBERG_DIR_KEY = 'eisenberg_dir'

DEFAULT_GRID_PARAMS = {GRID_DIM_KEY: 96, GRID_SPACING_KEY: 0.75,
                       SHELL_WIDTH_KEY: 2.0, NX_KEY: 10}

def get_molecule(input_file):
    ifs = oechem.oemolistream(input_file)
    mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, mol)
    ifs.close()

    oechem.OEPerceiveResidues(mol)
    oechem.OECenter(mol)

    return mol

def get_eisenberg_grid(params, mol, grid_type='PHOBIC'):
    eisenberg_scale = collections.defaultdict(float)
    eisenberg_scale['ALA'] = 0.25; eisenberg_scale['CYS'] = 0.04; eisenberg_scale['PHE'] = 0.61;
    eisenberg_scale['ILE'] = 0.73; eisenberg_scale['LEU'] = 0.53; eisenberg_scale['PRO'] = -0.07;
    eisenberg_scale['VAL'] = 0.54; eisenberg_scale['TRP'] = 0.37; eisenberg_scale['TYR'] = 0.02;
    eisenberg_scale['ASP'] = -0.72; eisenberg_scale['GLU'] = -0.62; eisenberg_scale['GLY'] = 0.16;
    eisenberg_scale['HIS'] = -0.40; eisenberg_scale['LYS'] = -1.1; eisenberg_scale['MET'] = 0.26;
    eisenberg_scale['ASN'] = -0.64; eisenberg_scale['GLN'] = -0.69; eisenberg_scale['ARG'] = -1.8;
    eisenberg_scale['SER'] = -0.26; eisenberg_scale['THR'] = -0.18;

    mol_copy = oechem.OEGraphMol(mol)
    for atom in mol_copy.GetAtoms():
        res = oechem.OEAtomGetResidue(atom)
        aa = res.GetName()
        if grid_type == 'PHOBIC' and eisenberg_scale[aa] < 0.0:
            mol_copy.DeleteAtom(atom)
            continue
        if grid_type == 'PHILIC' and eisenberg_scale[aa] > 0.0:
            mol_copy.DeleteAtom(atom)
            continue

        atom.SetRadius(3*np.abs(eisenberg_scale[aa]))

    mol_copy.Sweep()
    print(grid_type, mol_copy.NumAtoms(), oechem.OECount(mol_copy, oechem.OEIsHydrogen()))
    grid_spacing = params[GRID_SPACING_KEY]
    grid_dim = params[GRID_DIM_KEY]
    oe_grid = oegrid.OEScalarGrid(grid_dim, grid_dim, grid_dim, 0.0, 0.0, 0.0, grid_spacing)
    oegrid.OEMakeMolecularGaussianGrid(oe_grid, mol_copy)

    return oe_grid


def gen_eisenberg_array(params):
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

    phobic_grid = get_eisenberg_grid(params, mol, 'PHOBIC')
    philic_grid = get_eisenberg_grid(params, mol, 'PHILIC')

    grid_size = grid.GetSize()
    arr = np.zeros(grid_size)
    phobic_arr = np.zeros(grid_size)
    philic_arr = np.zeros(grid_size)
    idx = 0
    for i in range(0, grid_dim):
        for j in range(0, grid_dim):
            for k in range(0, grid_dim):
                v = surf_grid.GetValue(i, j, k)
                if 0 <= v < shell_width:
                    val = grid.GetValue(i, j, k)
                    arr[idx] = val
                    val = phobic_grid.GetValue(i, j, k)
                    phobic_arr[idx] = val
                    val = philic_grid.GetValue(i, j, k)
                    philic_arr[idx] = val

                idx += 1

    arr3d_esp = np.reshape(arr, (grid_dim, grid_dim, grid_dim, 1))
    arr3d_phobic = np.reshape(phobic_arr, (grid_dim, grid_dim, grid_dim, 1))
    arr3d_philic = np.reshape(philic_arr, (grid_dim, grid_dim, grid_dim, 1))

    return arr3d_esp, arr3d_phobic, arr3d_philic, mol


def prepare_cnn_input(df, args, train=True):
    hm_model_dir = args[HOMOLOGY_MODEL_DIR_KEY]
    if hm_model_dir is None:
        raise Exception('Homology model directory not specified')

    X = []
    y = []
    for row_idx, row in df.iterrows():
        entity = row[ENTITY_KEY]

        mol_file = os.path.join(hm_model_dir, entity + '.mol2')
        if len(args[EISENBERG_DIR_KEY]) > 0:
            esp_grids = get_eisenberg_grids(args, mol_file)
        else:
            esp_grids = generate_eisenberg_grids(args, mol_file)

        if args['num_channels'] == 3:
            combined_grid = [np.concatenate([esp_arr, phobic_arr, philic_arr], axis=0)
                             for esp_arr, phobic_arr, philic_arr, _ in esp_grids]
        else:
            combined_grid = [np.concatenate([phobic_arr, philic_arr], axis=0)
                             for _, phobic_arr, philic_arr, _ in esp_grids]

        X.extend(combined_grid)
        if train:
            log_visc = np.log10(row[VISCOSITY_KEY])
            y.extend([log_visc] * args[NX_KEY])
        else:
            y.extend([0.0] * args[NX_KEY])

    return np.array(X), np.array(y)


def get_eisenberg_grids(args, mol_file):
    eisenberg_dir = args[EISENBERG_DIR_KEY]
    eisenberg_array_output = []
    for i in range(args[NX_KEY]):
        with open('%s/rotation_%d/%s.pyb' % (eisenberg_dir, i + 1,
                                             os.path.basename(mol_file).split('.mol2')[0]), 'rb') as fptr:

            mol = get_molecule(os.path.join(os.path.join(eisenberg_dir, 'rotation_%d' % (i+1)), os.path.basename(mol_file)))
            esp_arr, phobic_arr, philic_arr = pickle.load(fptr)
            eisenberg_array_output.append((esp_arr, phobic_arr, philic_arr, mol))

    return eisenberg_array_output


def generate_eisenberg_grids(args, mol_file):
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
    eisenberg_array_output = p.map(gen_eisenberg_array, params)
    p.close()

    output = [(np.moveaxis(esp_array, 3, 0), np.moveaxis(phobic_array, 3, 0), np.moveaxis(philic_array, 3, 0), output_mol)
              for esp_array, phobic_array, philic_array, output_mol in eisenberg_array_output]
    return output


def prepare_training_input(df, args):
    return prepare_cnn_input(df, args, train=True)


def prepare_test_input(df, args):
    return prepare_cnn_input(df, args, train=False)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

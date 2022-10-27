import os
import pandas as pd
import glob
import argparse

import numpy as np
import torch

from openeye import oechem
from openeye import oegrid

from model import ViscosityNet
from utils import seed_everything
from utils import GRID_DIM_KEY, GRID_SPACING_KEY, ESP_DIR_KEY, HOMOLOGY_MODEL_DIR_KEY
from utils import get_molecule, calculate_attribution_grid
from utils import get_esp_grids, generate_esp_grids
from base import ENTITY_KEY



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


def overlay(reference_mol, fit_mol, attribution_mol=None):
    alignment = oechem.OEGetAlignment(reference_mol, fit_mol)
    rot = oechem.OEDoubleArray(9)
    trans = oechem.OEDoubleArray(3)
    oechem.OERMSD(reference_mol, fit_mol, alignment, True, True, rot, trans)
    oechem.OERotate(fit_mol, rot)
    oechem.OETranslate(fit_mol, trans)

    if attribution_mol is not None:
        oechem.OERotate(attribution_mol, rot)
        oechem.OETranslate(attribution_mol, trans)


def get_attribution_mol(args, attribution_grid):
    attribution_mol = oechem.OEGraphMol()
    grid_dim, grid_spacing = args[GRID_DIM_KEY], args[GRID_SPACING_KEY]
    significant_thres = args['significant_attribution_threshold']
    grid = oegrid.OEScalarGrid(grid_dim, grid_dim, grid_dim, 0.0, 0.0, 0.0, grid_spacing)
    for i in range(grid_dim):
        for j in range(grid_dim):
            for k in range(grid_dim):
                gradient = attribution_grid[0][0][i][j][k]

                if np.abs(attribution_grid[0][0][i][j][k]) > significant_thres:
                    x, y, z = grid.GridIdxToSpatialCoord(i, j, k)
                    if gradient > 0.0:
                        atom = attribution_mol.NewAtom(oechem.OEElemNo_O)
                    else:
                        atom = attribution_mol.NewAtom(oechem.OEElemNo_N)
                    atom.SetPartialCharge(attribution_grid[0][0][i][j][k])
                    attribution_mol.SetCoords(atom, oechem.OEFloatArray([x, y, z]))

    return attribution_mol


def generate_attributions(args, models):
    def save_molecule(f, mol):
        ofs = oechem.oemolostream(f)
        oechem.OEWriteMolecule(ofs, mol)
        ofs.close()

    reference_mol = get_molecule(args['reference_structure_file'], perceive_residue=True, center_mol=False)

    df = pd.read_csv(args['test_data_file'])

    hm_model_dir = args[HOMOLOGY_MODEL_DIR_KEY]
    output_attribution_dir = args['output_attribution_dir']
    for row_idx, row in df.iterrows():
        if args['process_structure_index'] >= 0 and args['process_structure_index'] != row_idx:
            continue
        mol_file = os.path.join(hm_model_dir, row[ENTITY_KEY] + '.mol2')
        if len(args[ESP_DIR_KEY]) > 0:
            esp_grids = get_esp_grids(args, mol_file)
        else:
            esp_grids = generate_esp_grids(args, mol_file)

        for grid_idx, (esp_grid, mol) in enumerate(esp_grids):
            for model_idx, model in enumerate(models):
                print('processing... row_idx: %d grid_idx: %d, model_idx: %d'
                      % (row_idx, grid_idx, model_idx))
                mol2 = oechem.OEGraphMol(mol)
                oechem.OEPerceiveResidues(mol2)
                attribution_grid, _ = calculate_attribution_grid(model, esp_grid, device)
                attribution_mol = get_attribution_mol(args, attribution_grid)
                overlay(reference_mol, mol2, attribution_mol)
                outfile_base = os.path.join(output_attribution_dir,
                                           '%s_%d_%d' % (row[ENTITY_KEY], grid_idx, model_idx))
                save_molecule(outfile_base + '.mol2', mol2)
                save_molecule(outfile_base + '.oeb.gz', attribution_mol)
                if len(args[ESP_DIR_KEY]) > 0:
                    pdb_file = mol_file.split('.mol2')[0] + '.pdb'
                    pdb_mol = get_molecule(pdb_file, perceive_residue=False, center_mol=False)
                    oechem.OEPerceiveResidues(mol2)
                    overlay(mol2, pdb_mol)
                    save_molecule(outfile_base + '.pdb', pdb_mol)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate attributions using PfAbNet models')
    parser.add_argument('--test_data_file', type=str, help='test set csv files with entity names')
    parser.add_argument('--reference_structure_file', type=str,
                        help='align each generated attribution molecule to the reference molecule (.mol2)')
    parser.add_argument('--homology_model_dir', type=str, help='homology model directory')
    parser.add_argument('--PfAbNet_model_prefix', type=str, default='PfAbNet', help='PfAbNet model prefix')
    parser.add_argument('--PfAbNet_model_dir', type=str, help='PfAbNet model directory')
    parser.add_argument('--grid_dim', type=int, default=96,
                        help='number of grid points along each axis (default = 96)')
    parser.add_argument('--grid_spacing', type=float, default=0.75,
                        help='spacing between grid points (default = 0.75 Angstrom)')
    parser.add_argument('--shell_width', type=float, default=2.0,
                        help='thickness of the surface shell (default 2.0 Angstrom)')
    parser.add_argument('--NX', type=int, default=10,
                        help='number of rotated structures for each input structure (default 10)')
    parser.add_argument('--processors', type=int, default=5,
                        help='Number of CPUs for ESP grid calculation (default 5)')
    parser.add_argument('--esp_dir', type=str, default='', help='directory with precomputed ESP grids')
    parser.add_argument('--significant_attribution_threshold', type=float,
                        help='significant attribution threshold')
    parser.add_argument('--process_structure_index', type=int, default=-1,
                        help='process structure index (default: -1, process all')
    parser.add_argument('--output_attribution_dir', type=str, help='directory to save attribution outputs')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()

    seed_everything(42)

    model_files_prefix = os.path.join(args.PfAbNet_model_dir, args.PfAbNet_model_prefix)
    model_files = glob.glob('%s*.pt' % model_files_prefix)

    args = vars(args)
    cnn_models = get_cnn_models(args, model_files)
    generate_attributions(args, cnn_models)




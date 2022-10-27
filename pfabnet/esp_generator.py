import os
import glob
import argparse
import pickle

import numpy as np
from openeye import oechem
from utils import generate_esp_grids



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate PfAbNet ESP grid input')
    parser.add_argument('--input_mols_dir', type=str, default='./',
                        help='directory containing antibody structures/models')
    parser.add_argument('--esp_output_dir', type=str, default='./',
                        help='directory to save the generated ESP grid files')
    parser.add_argument('--grid_dim', type=int, default=96,
                        help='number of grid points along each axis (default = 96)')
    parser.add_argument('--grid_spacing', type=float, default=0.75,
                        help='spacing between grid points (default = 0.75 Angstrom)')
    parser.add_argument('--shell_width', type=float, default=2.0,
                        help='thickness of the surface shell (default 2.0 Angstrom)')
    parser.add_argument('--NX', type=int, default=10,
                        help='augmentation level (default 10x)')
    parser.add_argument('--processors', type=int, default=10,
                        help='Number of CPUs for ESP grid calculation (default 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default 42)')

    parser.add_argument('-v', '--verbose', action='count', default=0)
    in_args = parser.parse_args()

    input_mols_dir = in_args.input_mols_dir
    esp_dir = in_args.esp_output_dir
    seed = in_args.seed

    args = in_args.__dict__
    np.random.seed(seed)

    try:
        os.mkdir(esp_dir)
    except Exception as e:
        pass

    mol_files = glob.glob(input_mols_dir + '/*.mol2')
    for mol_file in mol_files:
        print(mol_file)
        output = generate_esp_grids(args, mol_file)
        for idx, (esp_grid, output_mol) in enumerate(output):
            output_dir = os.path.join(esp_dir, 'rotation_%d' % (idx + 1))
            try:
                os.mkdir(output_dir)
            except Exception as e:
                pass

            base_mol_file = os.path.basename(mol_file).split('.mol2')[0]
            esp_file = os.path.join(output_dir, base_mol_file + '.pyb')
            with open(esp_file, 'wb') as fptr:
               pickle.dump(esp_grid, fptr)

            output_mol_file = os.path.join(output_dir, os.path.basename(mol_file))

            ofs = oechem.oemolostream(output_mol_file)
            oechem.OEWriteConstMolecule(ofs, output_mol)
            ofs.close()





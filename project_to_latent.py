from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys

from molecules.model import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset



def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='File of latent representation tensors for decoding.')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    parser.add_argument('--save_h5', type=str, help='Name of a file to write HDF5 output to.')

    return parser.parse_args()


def encode(args, model):
    data_train, data_test, charset, property_train, property_test = load_dataset(args.data)

    latent_dim = 292

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size=latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    x_latent = model.encoder.predict(data_train)
    if args.save_h5:
        h5f = h5py.File(args.save_h5, 'w')
        h5f.create_dataset('charset', data=charset)
        h5f.create_dataset('latent_vectors', data=x_latent)
        h5f.create_dataset('properties', data=property_train)
        h5f.close()
    else:
        np.savetxt(sys.stdout, x_latent, delimiter='\t')

def main():
    args = get_arguments()
    model = MoleculeVAE()

    encode(args, model)


if __name__ == '__main__':
    main()

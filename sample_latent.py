from __future__ import print_function

import argparse
import os, sys
import numpy as np

from molecules.model import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


LATENT_DIM = 292
PCA_COMPONENTS = 50


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='HDF5 file to read input data from.')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--save_location', type=str, metavar='N', default='',
                        help='File path where to save the plot.')
    parser.add_argument('--pca_components', type=int, metavar='N', default=PCA_COMPONENTS,
                        help='Number of PCA components to use')
    parser.add_argument('--color_bar', dest='color_bar', action='store_true',
                        help='If this flag is used, a vertical color bar will be included in the plot.')
    parser.set_defaults(color_bar=False)

    return parser.parse_args()


def visualize_latent_rep(args, x_latent, properties):
    print("Computing PCA")
    pca = PCA(n_components = args.pca_components)
    x_latent = pca.fit_transform(x_latent)

    if args.save_location == '':
        figs_path = 'figs/' + args.model.split('/')[-1].split('.h5')[0] + '.pdf'
    else:
        if not args.save_location.endswith('.pdf'):
            figs_path = args.save_location + '.pdf'
        else:
            figs_path = args.save_location

    directory = os.path.dirname(figs_path)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    if args.color_bar:
        fig = plt.figure(figsize=(7, 6))
    else:
        fig = plt.figure(figsize=(6, 6))

    cax = plt.scatter(x_latent[:, 0], x_latent[:, 1], c=properties, marker='.', s=0.1)

    if args.color_bar:
        cbar = fig.colorbar(cax, ticks=[np.amin(properties), np.amax(properties)])
        cbar.ax.set_yticklabels([str(np.amin(properties)), str(np.amax(properties))])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(figs_path, bbox_inches='tight')
    plt.show()


def main():
    args = get_arguments()
    model = MoleculeVAE()

    data, data_test, charset, properties_train, properties_test = load_dataset(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    x_latent = model.encoder.predict(data_test)

    visualize_latent_rep(args, x_latent, properties_test)

if __name__ == '__main__':
    main()

from __future__ import print_function

import matplotlib

matplotlib.use('Agg')

import argparse
import os
import numpy as np
import h5py

from molecules.model import MoleculeVAE
from molecules.utils import load_dataset

from pylab import figure, axes, scatter, title, show, savefig

from keras.utils.visualize_util import plot
from sklearn.manifold import TSNE

OUTFILE_NAME = 'image.png'

LATENT_DIM = 292
PCA_COMPONENTS = 50
TSNE_LEARNING_RATE = 750.0
TSNE_ITERATIONS = 1000
TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30.0

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--data', type=str, help='Path for data file to read')
    parser.add_argument('--model', type=str, help='Path for model file to visualize')
    parser.add_argument('--outfile', type=str, default=OUTFILE_NAME,
                        help = "Filename to write out. Default: %s" % OUTFILE_NAME)

    return parser.parse_args()

def visualize_model(args):
    model = MoleculeVAE()

    data, charset = load_dataset(args.data, split = False)

    if os.path.isfile(args.model):
        model.load(charset, args.model)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    plot(model.autoencoder, to_file = args.outfile)

def plot_2d(args):
    data = np.loadtxt(args.data, delimiter='\t')
    figure(figsize=(6, 6))
    scatter(data[:, 0], data[:, 1], marker = '.', linewidth = '0', s = 0.2)
    savefig(args.outfile, bbox_inches = 'tight')


def plot_3d(args):
    h5f = h5py.File(args.data, 'r')

    latent_rep = h5f['latent_vectors'][:]
    properties = h5f['properties'][:]

    tsne = TSNE(n_components = TSNE_COMPONENTS,
                perplexity = TSNE_PERPLEXITY,
                learning_rate = TSNE_LEARNING_RATE,
                n_iter = TSNE_ITERATIONS,
                verbose = 4)

    # x_latent_proj = tsne.fit_transform(latent_rep)

    figure(figsize=(6, 6))

    scatter(latent_rep[:, 0], latent_rep[:, 1], c=properties)
    savefig(args.outfile, bbox_inches='tight')



def main():
    args = get_arguments()

    if args.model:
        visualize_model(args)

    elif args.data:
        plot_2d(args)

if __name__ == '__main__':
    main()

import argparse
import os
from keras.utils.visualize_util import plot
from molecules.model import MoleculeVAE
from molecules.utils import load_dataset

LATENT_DIM = 292


def get_arguments():
    parser = argparse.ArgumentParser(description='LogP Optimizer')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--save_location', type=str, metavar='N', default='',
                        help='File path where to save the plot.')

    return parser.parse_args()


def visualize(args, model):
    if args.save_location == '':
        file_path = 'images/model.png'
    else:
        if not args.save_location.endswith('.png'):
            file_path = args.save_location + '.png'
        else:
            file_path = args.save_location

    directory = os.path.dirname(file_path)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    plot(model.autoencoder, file_path)


def main():
    args = get_arguments()

    data_train, data_test, charset, property_train, property_test = load_dataset(args.data)

    model = MoleculeVAE()
    model.create(charset, latent_rep_size=args.latent_dim, predictor='regression')

    visualize(args, model)


main()

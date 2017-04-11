import argparse
import numpy as np
from molecules.model import MoleculeVAE
from molecules.utils import load_dataset

LATENT_DIM = 292


def get_arguments():
    parser = argparse.ArgumentParser(description='Accuracy checker')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')

    return parser.parse_args()


def check_accuracy(original_smiles, reconstructed_smiles):
    N = original_smiles.shape[0]
    correctly_predicted = np.sum(np.all(original_smiles == reconstructed_smiles, axis=(1, 2)))

    return correctly_predicted / float(N)


def reconstruct_smiles(original_smiles, model):
    x_latent = model.encoder.predict(original_smiles)

    reconstructed_smiles = model.decoder.predict(x_latent)

    return reconstructed_smiles


def main():
    args = get_arguments()

    data_train, data_test, charset, property_train, property_test = load_dataset(args.data)

    model = MoleculeVAE()
    model.load(charset, args.model, latent_rep_size=args.latent_dim)

    reconstructed_smiles = reconstruct_smiles(data_train, model)

    accuracy = check_accuracy(data_train, reconstructed_smiles)

    print("Accuracy: " + str(accuracy))


if __name__ == '__main__':
    main()

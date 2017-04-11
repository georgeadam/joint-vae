import argparse
import numpy as np
from molecules.model import MoleculeVAE
from molecules.utils import load_dataset

LATENT_DIM = 292
np.set_printoptions(threshold=np.nan)

def get_arguments():
    parser = argparse.ArgumentParser(description='Accuracy checker')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--small', dest='small', action='store_true',
                        help='Use a random sample of 1000 molecules to test.')

    return parser.parse_args()


def check_accuracy(original_smiles, reconstructed_smiles):
    N = original_smiles.shape[0]
    correctly_predicted = np.sum(np.all(original_smiles == reconstructed_smiles, axis=(1, 2)))

    return correctly_predicted / float(N)


def reconstruct_smiles(original_smiles, model):
    x_latent = model.encoder.predict(original_smiles)

    reconstructed_smiles = model.decoder.predict(x_latent)

    a1, a2 = np.indices((original_smiles.shape[0], original_smiles.shape[1]))

    a_max = np.argmax(reconstructed_smiles, axis=2)

    reconstructed_smiles = np.zeros(reconstructed_smiles.shape)
    reconstructed_smiles[a1, a2, a_max] = 1

    return reconstructed_smiles


def main():
    args = get_arguments()

    data_train, data_test, charset, property_train, property_test = load_dataset(args.data)
    print(data_train.shape)
    if args.small:
        idx = np.random.randint(0, data_test.shape[0], size=1000)
        data_test = data_test[idx]

    model = MoleculeVAE()
    model.load(charset, args.model, latent_rep_size=args.latent_dim)

    reconstructed_smiles = reconstruct_smiles(data_test, model)

    accuracy = check_accuracy(data_test, reconstructed_smiles)

    print("Accuracy: " + str(accuracy))



if __name__ == '__main__':
    main()

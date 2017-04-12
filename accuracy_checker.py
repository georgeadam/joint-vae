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
    parser.add_argument('--small', dest='small', action='store_true',
                        help='Use a random sample of 1000 molecules to test.')
    parser.set_defaults(small=False)

    return parser.parse_args()


def check_true_accuracy(original_smiles, reconstructed_smiles):
    N = original_smiles.shape[0]
    correctly_predicted = np.sum(np.all(original_smiles == reconstructed_smiles, axis=(1, 2)))

    return correctly_predicted / float(N)


def check_ce_accuracy(original_smiles, property, model):
    optim_pred_loss, decoded_mean_acc = model.autoencoder.evaluate(original_smiles, [original_smiles, property], verbose=1)[2:4]
    return optim_pred_loss, decoded_mean_acc

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
    np.random.seed(100)
    data_train, data_test, charset, property_train, property_test = load_dataset(args.data)

    if args.small:
        idx = np.random.randint(0, data_test.shape[0], size=1000)
        data_test = data_test[idx]
        property_test = property_test[idx]

    model = MoleculeVAE()
    model.load(charset, args.model, latent_rep_size=args.latent_dim)

    reconstructed_smiles = reconstruct_smiles(data_test, model)

    true_accuracy = check_true_accuracy(data_test, reconstructed_smiles)
    optim_pred_loss, decoded_mean_acc = check_ce_accuracy(data_test, property_test, model)

    print("True Accuracy: " + str(true_accuracy))
    print('optim_pred_loss: ' + str(optim_pred_loss))
    print('decoded_mean_acc: ' + str(decoded_mean_acc))

if __name__ == '__main__':
    main()

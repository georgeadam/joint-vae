import argparse
import autograd.numpy as np
from autograd import grad
import os

from molecules.model import MoleculeVAE
from molecules.utils import load_dataset, from_one_hot_array, decode_smiles_from_indexes


LATENT_DIM = 292


def get_arguments():
    parser = argparse.ArgumentParser(description='LogP Optimizer')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to load the saved model from')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')

    return parser.parse_args()


def inspect_weights(model):
    print('Layer types:')
    print(model.layers[0])
    print(model.layers[1])
    print(model.layers[2])

    print('Layer Weights:')
    print(model.layers[0].get_weights())
    print(model.layers[1].get_weights())
    print(model.layers[2].get_weights())


def get_weights_for_model(model):
    W_1, b_1 = model.layers[1].get_weights()
    W_2, b_2 = model.layers[2].get_weights()

    return W_1, b_1, W_2, b_2


def compute_prediction_autograd(W_1, b_1, W_2, b_2, z):
    h = np.dot(W_1.T, z) + b_1

    y_pred = np.dot(W_2.T, h) + b_2

    return y_pred


def compute_prediction_keras(model, latent_rep):
    prediction = model.predictor.predict(latent_rep)

    return prediction


def get_latent_rep(model, molecule):
    latent_rep = model.encoder.predict(molecule)

    return latent_rep


def sgd(W_1, b_1, W_2, b_2, z, learning_rate=0.001, epochs = 100, print_every=50):
    deriv = grad(compute_prediction_autograd, 4)
    for epoch in range(epochs):
        dz = deriv(W_1, b_1, W_2, b_2, z)
        z = z + learning_rate * dz

        if epoch % print_every == 0:
            print('LogP: ' + str(compute_prediction_autograd(W_1, b_1, W_2, b_2, z)))

    return z


def main():
    args = get_arguments()
    np.random.seed(101)

    data_train, data_test, charset, property_train, property_test = load_dataset(args.data)
    model = MoleculeVAE()

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim)

    # Choose the first molecule from the training set as the molecule of interest
    molecule = np.array([data_train[0]])
    molecule_string = decode_smiles_from_indexes(map(from_one_hot_array, data_train[0]), charset)

    print('SMILES string: ' + molecule_string)

    # Get the numpy weight arrays for the linear predictor, and get the latent space representation of the molecule
    W_1, b_1, W_2, b_2 = get_weights_for_model(model.predictor)
    z = get_latent_rep(model, molecule)

    # Predict LogP using keras
    keras_prediction = compute_prediction_keras(model, z)
    print('Keras prediction: ' + str(keras_prediction))

    # Predict LogP using matrix multiplication formulas
    autograd_prediction = compute_prediction_autograd(W_1, b_1, W_2, b_2, z[0])
    print('Numpy prediction: ' + str(autograd_prediction))

    # Get an optimized molecule in terms of LogP
    optimized_latent_rep = sgd(W_1, b_1, W_2, b_2, z[0], learning_rate=0.001, epochs=100)

    # Print out the SMILES string of the optimized model
    decoded_optimized = model.decoder.predict(optimized_latent_rep.reshape(1, LATENT_DIM)).argmax(axis=2)[0]
    optimized_string = decode_smiles_from_indexes(decoded_optimized, charset)
    print('Optimized string: ' + optimized_string)

if __name__ == '__main__':
    main()
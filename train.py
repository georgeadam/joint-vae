from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
from keras.callbacks import Callback
from molecules.model import LossHistoryDecodedMean, LossHistoryOptim

NUM_EPOCHS = 1
BATCH_SIZE = 600
LATENT_DIM = 292
RANDOM_SEED = 1337

class MyCallback(Callback):
    def __init__(self, alpha):
        self.alpha = alpha
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if self.alpha < 0.4 and epoch >= 2:
            self.alpha = self.alpha + 0.05

            print(self.alpha)


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
                        help='Seed to use to start randomizer for shuffling.')
    return parser.parse_args()

def main():
    args = get_arguments()
    np.random.seed(args.random_seed)

    from molecules.model import MoleculeVAE
    from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
        decode_smiles_from_indexes, load_dataset
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    
    data_train, data_test, charset, property_train, property_test = load_dataset(args.data)
    model = MoleculeVAE()
    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim)
    else:
        model.create(charset, latent_rep_size = args.latent_dim, predictor='regression')

    checkpointer = ModelCheckpoint(filepath = args.model,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    tbCallBack = TensorBoard(log_dir='./graph')

    optimHistory =LossHistoryOptim()
    decodedHistory = LossHistoryDecodedMean()
    # Notice how there are two different desired outputs. This is due to the fact that our model has 2 outputs,
    # namely the output of the decoder, and the output of the property prediction module.
    vae_weight = 1.0
    optim_weight = 0.1

    model.autoencoder.compile(optimizer='Adam',
                             loss=[model.vae_loss, model.predictor_loss],
                             metrics=['accuracy'],
                             loss_weights=[vae_weight, optim_weight])

    model.autoencoder.fit(
        data_train, # This is our input
        {'decoded_mean': data_train, 'optim_pred': property_train}, # These are the two desired outputs
        shuffle = True,
        nb_epoch = args.epochs,
        batch_size = args.batch_size,
        callbacks = [checkpointer, reduce_lr, tbCallBack, optimHistory, decodedHistory],
        validation_data = (data_test, {'decoded_mean': data_test, 'optim_pred': property_test})
    )

if __name__ == '__main__':
    main()

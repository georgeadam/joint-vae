from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
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


def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + \
               model_name + "-{epoch:02d}-{val_decoded_mean_acc:.2f}-{val_optim_pred_loss:.2f}-{val_loss:.2f}.h5"
    directory = os.path.dirname(filepath)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=False)

    return checkpointer


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
    parser.add_argument('--schedule', dest='schedule', action='store_true',
                        help='Indicates whether or not to train using a schedule for the loss weights, else' +
                             'first train a VAE for args.epochs, and then focus on the prediction module for' +
                        'another args.epochs. ')
    parser.add_argument('--vae', dest='vae', action='store_true',
                        help='Indicates whether or not to train using just the VAE error.')
    parser.add_argument('--optim', dest='optim', action='store_true',
                        help='Indicates whether or not to train focusing on the optimization error.')
    parser.set_defaults(schedule=False)
    parser.set_defaults(vae_only=False)
    parser.set_defaults(vae_optim=False)
    return parser.parse_args()

def main():
    args = get_arguments()
    np.random.seed(args.random_seed)

    from molecules.model import MoleculeVAE
    from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
        decode_smiles_from_indexes, load_dataset
    from keras.callbacks import ReduceLROnPlateau
    
    data_train, data_test, charset, property_train, property_test = load_dataset(args.data)
    model = MoleculeVAE()
    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim)
    else:
        model.create(charset, latent_rep_size = args.latent_dim, predictor='regression')

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    # Notice how there are two different desired outputs. This is due to the fact that our model has 2 outputs,
    # namely the output of the decoder, and the output of the property prediction module.
    if args.schedule:
        checkpointer = create_model_checkpoint('schedule', 'model_schedule')

        vae_weight = 1.0
        optim_weight_schedule = [0.10, 0.08, 0.08, 0.18, 0.24, 0.30, 0.40, 0.45, 0.6, 0.8, 0.9, 1.0,
                                 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.1, 2.2,
                                 2.0, 2.4, 2.6, 2.8, 3.0]

        for epoch in range(args.epochs):
            print("Optim weight: " + str(optim_weight_schedule[epoch]))
            model.autoencoder.compile(optimizer='Adam',
                                     loss=[model.vae_loss, model.predictor_loss],
                                     metrics=['accuracy'],
                                     loss_weights=[vae_weight, optim_weight_schedule[epoch]])

            model.autoencoder.fit(
                data_train,  # This is our input
                {'decoded_mean': data_train, 'optim_pred': property_train},  # These are the two desired outputs
                shuffle=True,
                nb_epoch=1,
                batch_size=args.batch_size,
                callbacks=[checkpointer, reduce_lr],
                validation_data=(data_test, {'decoded_mean': data_test, 'optim_pred': property_test}))
    else:
        if args.vae:
            checkpointer = create_model_checkpoint('vae_only', 'model_vae_only')

            model.autoencoder.compile(optimizer='Adam',
                                      loss=[model.vae_loss, model.predictor_loss],
                                      metrics=['accuracy'],
                                      loss_weights=[1.0, 0.0])

            model.autoencoder.fit(
                data_train, # This is our input
                {'decoded_mean': data_train, 'optim_pred': property_train}, # These are the two desired outputs
                shuffle = True,
                nb_epoch = args.epochs,
                batch_size = args.batch_size,
                callbacks = [checkpointer, reduce_lr],
                validation_data = (data_test, {'decoded_mean': data_test, 'optim_pred': property_test})
            )

        if args.optim:
            checkpointer = create_model_checkpoint('vae_optim', 'model_vae_optim')

            model.autoencoder.compile(optimizer='Adam',
                                      loss=[model.vae_loss, model.predictor_loss],
                                      metrics=['accuracy'],
                                      loss_weights=[0.001, 20.0])

            model.autoencoder.fit(
                data_train, # This is our input
                {'decoded_mean': data_train, 'optim_pred': property_train}, # These are the two desired outputs
                shuffle = True,
                nb_epoch = args.epochs,
                batch_size = args.batch_size,
                callbacks = [checkpointer, reduce_lr],
                validation_data = (data_test, {'decoded_mean': data_test, 'optim_pred': property_test})
            )


if __name__ == '__main__':
    main()

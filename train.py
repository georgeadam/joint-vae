from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
from keras.callbacks import Callback
from molecules.model import LossHistoryDecodedMean, LossHistoryOptim

NUM_EPOCHS = 1
BATCH_SIZE = 1000
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
    parser.add_argument('--kl_weight', type=float, metavar='N', default=0)
    parser.add_argument('--opt_weight', type=float, metavar='N', default=0)
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

    filepath = "model_linear-schedule-{epoch:02d}-{val_optim_pred_loss:.2f}-{val_decoded_mean_acc:.2f}.h5"
    checkpointer = ModelCheckpoint(filepath = filepath,
                                   verbose = 1,
                                   save_best_only = False)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    #tbCallBack = TensorBoard(log_dir='./graph')

    optimHistory =LossHistoryOptim()
    decodedHistory = LossHistoryDecodedMean()
    # Notice how there are two different desired outputs. This is due to the fact that our model has 2 outputs,
    # namely the output of the decoder, and the output of the property prediction module.
    kl_weight = [0.55, 0.55, 0.55, 0.55, 0.55, 0.65, 0.65, 0.70, 0.8, 0.8]
    opt_weight = [0.15, 0.15, 0.15, 0.20, 0.25, 0.25, 0.30, 0.30, 0.3, 0.3] 
    for epoch in range(args.epochs):
      print("KL weight: " + str(kl_weight[epoch]))
      model.autoencoder.compile(optimizer='Adam',
                               loss=[model.xent_loss, model.kl_loss, model.predictor_loss],
                               metrics=['accuracy'],
                               loss_weights=[1,kl_weight[epoch], opt_weight[epoch]])

      model.autoencoder.fit(
          data_train, # This is our input
          [ data_train,np.zeros([data_train.shape[0],1]),property_train], # These are the two desired outputs
          shuffle = True,
          nb_epoch = 1,
          batch_size = args.batch_size,
          #callbacks = [checkpointer, reduce_lr, tbCallBack, optimHistory, decodedHistory],
          callbacks = [checkpointer, reduce_lr, optimHistory, decodedHistory],
          validation_data = (data_test,[ data_test,np.zeros([data_test.shape[0],1]),property_test] )

    )

if __name__ == '__main__':
    main()

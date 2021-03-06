import copy
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
from keras.callbacks import Callback


class LossHistoryOptim(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('optim_pred_loss'))


class LossHistoryDecodedMean(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('decoded_mean_acc'))


class MoleculeVAE():
    autoencoder = None

    alpha = K.variable(0.1)

    def create(self,
               charset,
               max_length=120,
               latent_rep_size=292,
               weights_file=None,
               num_classes=2,
               predictor='regression'):
        charset_length = len(charset)

        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        self.autoencoder = Model(
            x1,
            [self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            ), self._buildRegressionPredictor(z1, latent_rep_size) if predictor == 'regression' else
                self._buildClassificationPredictor(z1, latent_rep_size, num_classes)]
        )

        self.vae_loss = vae_loss

        if predictor == 'regression':
            self.predictor = Model(
                encoded_input,
                self._buildRegressionPredictor(encoded_input, latent_rep_size)
            )
        else:
            self.predictor = Model(
                encoded_input,
                self._buildClassificationPredictor(encoded_input, latent_rep_size, num_classes)
            )

        if weights_file:
            self.autoencoder.load_weights(weights_file, by_name=True)
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)
            self.predictor.load_weights(weights_file, by_name=True)

        self.predictor_loss = 'mean_squared_error' if predictor == 'regression' else 'categorical_crossentropy'
        self.autoencoder.compile(optimizer='Adam',
                                 loss=[vae_loss, self.predictor_loss],
                                 metrics=['accuracy'],
                                 loss_weights=[0.001, 20])

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std=0.01):
        h = Convolution1D(9, 9, activation='relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation='relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., std=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences=True, name='gru_1')(h)
        h = GRU(501, return_sequences=True, name='gru_2')(h)
        h = GRU(501, return_sequences=True, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

    def _buildRegressionPredictor(self, z, latent_rep_size, prop='LogP'):
        h = Dense(latent_rep_size, name='optimz_h1', activation='linear')(z)
        # h = Dense(latent_rep_size, name='optimz_h2', activation='tanh')(h)
        # h = Dense(latent_rep_size, name='optimz_h3', activation='tanh')(h)
        return Dense(1, name='optim_pred', activation='linear')(h)

    def _buildClassificationPredictor(self, z, latent_rep_size, num_classes=2, prop='fda'):
        h = Dense(latent_rep_size, name='optimz_h1', activation='linear')(z)
        # h = Dense(latent_rep_size, name='optimz_h2', activation='tanh')(h)
        # h = Dense(latent_rep_size, name='optimz_h3', activation='tanh')(h)
        return Dense(num_classes, name='optim_pred', activation='softmax')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)

    def load(self, charset, weights_file, latent_rep_size=292, num_classes=2, predictor='regression'):
        self.create(charset, weights_file=weights_file, latent_rep_size=latent_rep_size, num_classes=num_classes,
                    predictor=predictor)

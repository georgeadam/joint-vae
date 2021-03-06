# Jointly training a VAE and prediction model on SMILES strings with Keras

<table style="border-collapse: collapse">
<tr>
<td style="vertical-align: top" valign="top">
    <strong>Abstract</strong>
    <p>How can pharmaceutical companies and medicinal chemists create new drug-like molecules or optimize existing
     ones in a cost-effective way? We introduce a training procedure for creating meaningful continuous latent-space 
     representations from discrete representations of molecules. By having a continuous latent-space representation 
     where molecules with similar values of a desired property are encouraged to be grouped together, this makes 
     optimization easier since less time is spent bouncing between local optima. Furthermore, we investigate several 
     approaches to achieving this goal by using dynamically weighted loss functions that emphasize different objectives 
     at the appropriate times during training. We show that using a carefully chosen training regime, it is possible to 
     create clear linear relationships between the continuous latent-space and desired drug-like properties. </p>
    <p>
        <strong>Link to previous works</strong><br />
        <a href="https://arxiv.org/abs/1610.02415">arXiv</a>
    </p>
</td><td width="506">
<img src="images/model.png" width="506" /></img>
</td>
</tr>
</table>

## Credits
The code in this project is inspired by and forked from the following repository
<https://github.com/maxhodak/keras-molecules>.

## Requirements

Install using `pip install -r requirements.txt` or build a docker container: `docker build .`

The docker container can also be built different TensorFlow binary, for example in order to use GPU:

`docker build --build-arg TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl .`

You'll need to ensure the proper CUDA libraries are installed for this version to work.

## Getting the datasets

A small 50k molecule dataset is included in `data/smiles_50k.h5` to make it easier to get started playing around with the model. A much larger 500k ChEMBL 21 extract is also included in `data/smiles_500k.h5`. A model trained on `smiles_500k.h5` is included in `data/model_500k.h5`.

All h5 files in this repo by [git-lfs](https://git-lfs.github.com/) rather than included directly in the repo. Use git lfs pull to get the data. 

To download original datasets to work with, you can use the `download_dataset.py` script:

* `python download_dataset.py --dataset zinc12`
* `python download_dataset.py --dataset chembl22`
* `python download_dataset.py --uri http://my-domain.com/my-file.csv --outfile data/my-file.csv`

## Preparing the data

To train the network you need a lot of SMILES strings. The `preprocess.py` script assumes you have an HDF5 file that contains a table structure, one column of which is named `structure` and contains one SMILES string no longer than 120 characters per row. The script then:

- Normalizes the length of each string to 120 by appending whitespace as needed.
- Builds a list of the unique characters used in the dataset. (The "charset")
- Substitutes each character in each SMILES string with the integer ID of its location in the charset.
- Converts each character position to a one-hot vector of len(charset).
- Saves this matrix to the specified output file.

Example:

`python preprocess.py data/smiles_50k.h5 data/processed.h5`

The output of the `preprocess.py` script depends heavily on the option --property_column. The outputted
h5py file will have a vector dedicated to storing the values found in the column name specified by this option. For example,
if we call ths script as follows:

`python preprocess.py data/smiles_50k.h5 data/processed.h5 --property_column="LogP"`

the LogP column will be stored in file `data/processed.h5` and can be accessed via the following code snippet:

```python
    h5f = h5py.File(filename, 'r')

    property_train = h5f['property_train'][:]
```

The property_train vector is then used as the supplementary output for the joint model, i.e. when jointly optimizing the
VAE loss and the property prediction network.
## Training the network

The preprocessed data can be fed into the `train.py` script. The first argument is the location of the preprocessed 
data, and the second argument is location of an existing model if there is one. An important flag when it comes to training
is `--schedule`. Using this flag makes the training procedure use a schedule of weights for the MSE loss. 
Models will be saved to the directory `schedule` when using this flag.

Omitting this flag requires the user to include some combination of the following flags: `--vae` or `--optim`.
Using just the `--vae` flag will train the model with a weight of 0 on the prediction loss for the specified 
number of epochs and save intermediate models to the directory `--vae_only`. 

Using just the `--optim` flag will train the model with weight mainly on the prediction loss for the specified number
of epochs and save the intermediate models to the directory `--vae_optim`.

Using both the `--vae` flag and the `--optim` flag will train the model as a pure VAE first for the specified number
of epochs, and then focus on the minimizing the prediction loss also for the specified number of epochs. 

Examples:

```
python train.py data/processed.h5 model.h5 --epochs 20 --schedule
python train.py data/processed.h5 model.h5 --epochs 20 --vae
python train.py data/processed.h5 model.h5 --epochs 20 --optim
python train.py data/processed.h5 model.h5 --epochs 20 --vae --optim
```

If a model file already exists it will be opened and resumed. If it doesn't exist, a new brand new model will be 
created. Keep in mind that models will still be saved to the directories mentioned in the 
above paragraphs.

By default, the latent space is 292-D per the paper, and is configurable with the `--latent_dim` flag. If you use a non-default latent dimensionality don't forget to use `--latent_dim` on the other scripts (eg `sample.py`) when you operate on that model checkpoint file or it will be confused.

## Visualizing the latent space
The `sample_latent.py` script is used to visualize the latent space projected onto the two
PCA dimensions accounting for the most variance. It displays the plot interactively and also saves it as
a PDF to the `figs/` directory. A particular path to save the plot to can be specified using the
`--save_location` flag. Note that even when specifying a location to save to, the file will result in a PDF,
regardless of the provided extension.

Examples:

```
python sample_latent.py data/processed.h5 vae_optim/best_model.h5
python sample_latent.py data/processed.h5 vae_optim/best_model.h5 --save_location="best/model_vis.pdf"
```

## Sampling from a trained model

The `sample.py` script can be used to either run the full autoencoder (for testing) or either the encoder or decoder halves using the `--target` parameter. The data file must include a charset field.

Examples:

```
python sample.py data/processed.h5 model.h5 --target autoencoder

python sample.py data/processed.h5 model.h5 --target encoder --save_h5 encoded.h5

python sample.py target/encoded.h5 model.h5 --target decoder
```

## Performance

After 20 epochs of training just a VAE on a 500,000 molecule extract (400,000 in training set, 100,000 in testing set) 
from ZINC 12 (~6 hours on a NVIDIA GTX 1070), a loss of 0.43 and a reconstruction accuracy of
0.98 was achieved. 

Projecting the dataset onto 2D latent space reaveals that the latent space forms a relatively homogenous region
consistenting of vertical bands with no clear direction of increasing LogP:

<img src="images/vae_only.png" />

After another 20 epochs of focusing mainly on LogP prediction, model acheived a prediction loss of 0.13,
but the reconstruction accuracy went down to 0.90,

Projecting the dataset onto 2D latent space shows that the space forms a much simpler manifold than training
just a pure VAE. There is now a clear increasing direction of LogP:

<img src="images/vae_optim.png" />

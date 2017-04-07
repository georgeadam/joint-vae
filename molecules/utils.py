import gzip
import pandas
import h5py
import numpy as np

def one_hot_array(i, n):
    return map(int, [ix == i for ix in xrange(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def load_dataset(filename, prop=True, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]

    if prop:
        property_train = h5f['property_train'][:]
        property_test = h5f['property_test'][:]

    h5f.close()

    if split:
        if prop:
            return (data_train, data_test, charset, property_train, property_test)

        return (data_train, data_test, charset)
    else:
        if prop:
            return (data_test, charset, property_train)

        return (data_test, charset)

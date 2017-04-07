import pandas
import numpy as np
from molecules.utils import load_dataset

ZINC_PATH = 'data/zinc12.h5'


def read_data():
    data = pandas.read_hdf(ZINC_PATH, 'table', start=0, stop=10)
    # data_train, data_test, charset, property_train, property_test = load_dataset(ZINC_PATH)

    print(data)
    # print(property_train[0:10])


def main():
    read_data()


if __name__ == '__main__':
    main()


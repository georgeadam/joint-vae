import argparse
import pandas
from molecules.utils import load_dataset


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')

    parser.add_argument('--file_path', type=str, metavar='N',
                        help='Seed to use to start randomizer for shuffling.')
    parser.add_argument('--processed', dest='processed', action='store_true',
                        help='Indicates whether the specified file has been processed or is a downloaded dataset.')
    return parser.parse_args()

def read_raw_dataset(file_path):
    data = pandas.read_hdf(file_path, 'table')

    print("First 10 rows: \n")
    print(data[:10])
    print("\n\n")
    print("Any null rows? " + str(data[pandas.isnull(data).any(axis=1)]))


def read_processed_dataset(file_path):
    data_train, data_test, charset, property_train, property_test = load_dataset(file_path)
    print("Property of first 10 molecules: ")
    print(property_train[0:10])


def main():
    args = get_arguments()
    if args.processed:
        read_processed_dataset(args.file_path)
    else:
        read_raw_dataset(args.file_path)


if __name__ == '__main__':
    main()


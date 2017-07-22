import numpy as np

from import_all import *

if __name__ == '__main__':
    train_data, train_weights, train_labels, test_data, *ret = import_from_csv(path='Datasets', drop_labels=True)


    # Find data split (s vs. b) in training set
    train_labels = np.array(train_labels.values, dtype='int32')
    label_counts = np.bincount(train_labels)

    print('\nTraining data has {} signal and {} background events ({:0.2f}%-{:0.2f}% split)'.
          format(label_counts[1], label_counts[0], label_counts[1] * 100.0 / (label_counts[0] + label_counts[1]),
                 label_counts[0] * 100.0 / (label_counts[0] + label_counts[1])))

    # Number of missing values
    print('\nTraining set has {}/{} ({:0.2f}%) missing values.'.
          format(np.sum(train_data.values == -999), np.prod(train_data.shape),
                 100 * np.sum(train_data.values == -999) / np.prod(train_data.shape)))

    print('\nTest set has {}/{} ({:0.2f}%) missing values.'.
          format(np.sum(test_data.values == -999), np.prod(train_data.shape),
                 100 * np.sum(test_data.values == -999) / np.prod(train_data.shape)))

    print(train_data.describe())
    train_data.describe().to_csv('Train_stats.csv')
    print(test_data.describe())
    test_data.describe().to_csv('Test_stats.csv')

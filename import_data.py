import os
import pickle
import pandas as pd


def import_from_csv(path='', drop_labels=True):

    """
    Method to import data from CSV files and split them into training and testing samples with associated labels
    and weights when available.
    :param path: path to CSV file
    :param drop_labels: Bool controlling whether the 'Label' column is preserved or dropped
    :return: returns train_data, weights, labels, test_data, train_event_id, test_event_id
    """
    train_data = pd.read_csv(os.path.join(path, 'training.csv'))
    test_data = pd.read_csv(os.path.join(path, 'test.csv'))

    columns = [c for c in train_data]

    weights = train_data[columns[-2]]
    labels = train_data[columns[-1]]
    train_event_id = train_data[columns[0]]
    test_event_id = test_data[columns[0]]

    f = lambda x: 1 if x == 's' else 0
    labels = labels.apply(f)

    print(columns)

    # drop event index, weights, and labels
    if drop_labels:
        train_data = train_data.drop(columns[-1], axis=1)

    train_data = train_data.drop(columns[-2], axis=1)
    train_data = train_data.drop(columns[0], axis=1)
    test_data = test_data.drop(columns[0], axis=1)

    return train_data, weights, labels, test_data, train_event_id, test_event_id


if __name__ == '__main__':
    train_data, train_weights, train_labels, test_data, *ret = import_from_csv(path='Datasets', drop_labels=True)
    print([c for c in test_data])

    with open('Datasets\data_orig.pkl', 'wb') as f:
        pickle.dump(dict(tr_data=train_data, test_data=test_data, test_ID=ret[1], w=train_weights, lbls=train_labels), f)

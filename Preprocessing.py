import pickle
import numpy as np
import pandas as pd

import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier

from import_data import import_from_csv

font = {'family': 'normal',
        'size': 28}

matplotlib.rc('font', **font)


def pre_impute(train_data, test_data):
    """
    Impute sets by replacing -999 by means
    """
    imp = Imputer(missing_values=-999, strategy="mean")
    imp.fit(train_data)
    train_data_imp = imp.fit_transform(train_data)
    train_data.iloc[:] = train_data_imp

    imp.fit(test_data)
    test_data_imp = imp.fit_transform(test_data)
    test_data.iloc[:] = test_data_imp

    return train_data, test_data


def pre_mi(train_data, test_data):
    """
    Apply feature selection using mutual information between feature-label pairs
    """
    mi = mutual_info_classif(train_data.values, train_labels, discrete_features='auto', n_neighbors=3)
    # mi = np.array([0.15027041, 0.08795945, 0.07692012, 0.02059609, 0.04814339, 0.04478382,
    #                0.04618069, 0.01617492, 0., 0.02626618, 0.04270419, 0.0467124,
    #                0.04156568, 0.05482185, 0.00723092, 0.00115859, 0.00296002, 0.01042242,
    #                0.0007641, 0.01487462, 0., 0.01879071, 0.02657806, 0.02617603,
    #                0.03124579, 0.02035788, 0.02574723, 0.03365665, 0.02540334, 0.0193679])

    ## PLOT GENERATION CODE
    # ax = sns.barplot(list(range(30)), mi)
    # ax.set(xlabel='Features', ylabel='Mutual Information')
    # ax.set_xticklabels([c for c in train_data])
    # plt.plot([-1,30], [0.04, 0.04], 'r')
    #
    # for item in ax.get_xticklabels():
    #     item.set_rotation(-90)
    # plt.tight_layout()
    # plt.show()

    filter = mi > 0.04 #predefined threshold to filter features
    train_data = train_data.iloc[:, filter]
    test_data = test_data.iloc[:, filter]

    print('\nMI selection:', [c for c in train_data])

    return train_data, test_data


def pre_ig(train_data, test_data, labels):
    """
    Apply feature selection using Information Gain
    NOTE: OBSOLETE
    """
    cla = DecisionTreeClassifier()
    cla = cla.fit(train_data, labels)

    ig = cla.feature_importances_

    ax = sns.barplot(list(range(30)), ig)
    ax.set(xlabel='Features', ylabel='Information Gain')
    ax.set_xticklabels([c for c in train_data])
    plt.plot([-1, 30], [0.04, 0.04], 'r')
    for item in ax.get_xticklabels():
        item.set_rotation(-90)
    plt.tight_layout()
    plt.show()

    filter = ig > 0.04
    train_data = train_data.iloc[:, filter]
    test_data = test_data.iloc[:, filter]

    print('\nIG selection:', [c for c in train_data])
    # IG selection: ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'PRI_tau_pt']
    # IG selection: ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_deltar_tau_lep', 'DER_met_phi_centrality', 'PRI_tau_pt'] IMP

    return train_data, test_data


def pre_pca(train_data, test_data, labels):
    """
    Apply feature reduction using Principal Component Analysis
    """
    pca = PCA(n_components=3)
    pca = pca.fit(train_data)
    pca_comp = pca.explained_variance_ratio_

    print('>>>>', pca.components_[:3, :])
    print(pca.components_[2])

    ## PLOT GENERATION CODE
    # ax = sns.barplot(list(range(30)), pca_comp)
    # ax.set(xlabel='PCA Component', ylabel='Explained Variance Ratio')
    # ax.set_xticklabels(['Comp#{}'.format(c) for c in np.arange(1, 30)])
    # for item in ax.get_xticklabels():
    #     item.set_rotation(-90)
    # plt.tight_layout()

    train_data_trans = pca.transform(train_data)

    ## MORE PLOT GENERATION CODE
    # fig, ax = plt.subplots(3, sharex=True)
    #
    # ax[0].bar(np.arange(30), pca.components_[0], align='center', alpha=0.5)
    # ax[0].set_xlim([-1, 30])
    # ax[0].set_ylabel('Comp #1')
    #
    # ax[1].bar(np.arange(30), pca.components_[1], align='center', alpha=0.5)
    # ax[1].set_xlim([-1, 30])
    # ax[1].set_ylabel('Comp #2')
    #
    # ax[2].bar(np.arange(30), pca.components_[2], align='center', alpha=0.5)
    # ax[2].set_xlim([-1, 30])
    # ax[2].set_ylabel('Comp #3')
    # ax[2].set_xticks(np.arange(30))
    # ax[2].set_xticklabels([c for c in test_data])
    # for item in ax[2].get_xticklabels():
    #     item.set_rotation(-90)
    # plt.gcf().subplots_adjust(bottom=0.3)
    # plt.show()

    test_data_trans = pca.transform(test_data)

    return pd.DataFrame(train_data_trans, columns=['PCA_Comp_1', 'PCA_Comp_2', 'PCA_Comp_3']), pd.DataFrame(
        test_data_trans, columns=['PCA_Comp_1', 'PCA_Comp_2', 'PCA_Comp_3'])


if __name__ == '__main__':
    train_data, train_weights, train_labels, test_data, *ret = import_from_csv(path='Datasets', drop_labels=True)

    pre_output = pre_impute(train_data, test_data)
    with open('Datasets\Data_orig_imp.pkl', 'wb') as f:
        pickle.dump(dict(tr_data=pre_output[0], test_data=pre_output[1], test_ID=ret[1], w=train_weights, lbls=train_labels),
                    f)

    ############### FEATURE SELECTION: DERIVED #####################

    with open('Datasets\Data_DER.pkl', 'wb') as f:
        pickle.dump(dict(tr_data=train_data.iloc[:, :13], test_data=test_data.iloc[:, :13], test_ID=ret[1], w=train_weights, lbls=train_labels),
                    f)

    pre_output = pre_impute(train_data.iloc[:, :13], test_data.iloc[:, :13])
    with open('Datasets\Data_DER_imp.pkl', 'wb') as f:
        pickle.dump(dict(tr_data=pre_output[0], test_data=pre_output[1], test_ID=ret[1], w=train_weights, lbls=train_labels),
                    f)

    ############### FEATURE SELECTION: PRIMITIVE #####################

    with open('Datasets\Data_PRI.pkl', 'wb') as f:
        pickle.dump(dict(tr_data=train_data.iloc[:, 13:], test_data=test_data.iloc[:, 13:], test_ID=ret[1], w=train_weights, lbls=train_labels),
                    f)

    pre_output = pre_impute(train_data.iloc[:, 13:], test_data.iloc[:, 13:])
    with open('Datasets\Data_PRI_imp.pkl', 'wb') as f:
        pickle.dump(dict(tr_data=pre_output[0], test_data=pre_output[1], test_ID=ret[1], w=train_weights, lbls=train_labels),
                    f)

    ############### FEATURE SELECTION: MI #####################

    pre_output = pre_mi(train_data, test_data)
    with open('Datasets\Data_mi.pkl', 'wb') as f:
        pickle.dump(dict(tr_data=pre_output[0], test_data=pre_output[1], test_ID=ret[1], w=train_weights, lbls=train_labels), f)

    pre_output = pre_impute(train_data, test_data)
    pre_output = pre_mi(pre_output[0], pre_output[1])
    with open('Datasets\Data_mi_imp.pkl', 'wb') as f:
        pickle.dump(dict(tr_data=pre_output[0], test_data=pre_output[1], test_ID=ret[1], w=train_weights, lbls=train_labels), f)

    ############### FEATURE REDUCTION: PCA #####################

    pre_output = pre_pca(train_data, test_data, train_labels)
    with open('Datasets\Data_pca.pkl', 'wb') as f:
        pickle.dump(
            dict(tr_data=pre_output[0], test_data=pre_output[1], test_ID=ret[1], w=train_weights, lbls=train_labels), f)

    pre_output = pre_impute(train_data, test_data)
    pre_output = pre_pca(pre_output[0], pre_output[1], train_labels)
    with open('Datasets\Data_pca_imp.pkl', 'wb') as f:
        pickle.dump(
            dict(tr_data=pre_output[0], test_data=pre_output[1], test_ID=ret[1], w=train_weights, lbls=train_labels), f)

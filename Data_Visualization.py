import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler

from import_all import *

sns.set_context("paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20,
                             'xtick.labelsize': 20,
                             'ytick.labelsize': 20,
                             'legend.fontsize': 20})


def box_plot_data(data=None):
    """
    Method to generate box plot
    :param data: Pandas dataframe to be plotted
    """
    assert data is not None
    data2 = pd.melt(data, id_vars='Label')
    sns.boxplot(x='variable', y='value', hue='Label', vert=False, data=data2, showfliers=False)
    plt.show()
    plt.savefig('Figures/Boxplot.png')


if __name__ == "__main__":
    train_data, train_weights, train_labels, test_data, *ret = import_from_csv(path='Datasets', drop_labels=False)

    # subsample data to 10%
    frac_train_data = train_data.sample(frac=0.1)

    # Normalize data
    rs = RobustScaler()
    rs = rs.fit(train_data.iloc[:,:-1])
    train_data.iloc[:,:-1] = rs.transform(train_data.iloc[:,:-1])

    box_plot_data(data=train_data)
    print("plot complete")

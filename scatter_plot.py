"""
Scatter plot program that finds most similar features and shows a scatter  plot for them
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def scatter_plot():
    """
    Finds most similar features using correlation matrix and draws a scatter plot for them
    """
    df_data = pd.read_csv('datasets/dataset_train.csv', index_col='Index')
    df_data['Best Hand'].replace({'Left': 0, 'Right': 1}, inplace=True)
    df_plot = df_data.drop(['First Name', 'Last Name', 'Birthday'], axis=1)
    corrs = abs(df_plot.corr())
    np.fill_diagonal(corrs.values, 0)
    corrs = corrs.stack()
    most_similar = corrs.sort_values(ascending=False).index[0]
    plt.scatter(df_data[most_similar[0]], df_data[most_similar[1]])  # pylint: disable=E1136
    plt.xlabel(most_similar[0])
    plt.ylabel(most_similar[1])
    plt.savefig("scatter_plot.png")


if __name__ == '__main__':
    scatter_plot()

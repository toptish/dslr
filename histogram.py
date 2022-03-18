"""
Histogram program that helps to find similar distributions of classes among features
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from maths import min_val, max_val


def histogram():
    """
    Creates a histogram plot showing distributions among classes for all features
    """
    df_data = pd.read_csv('datasets/dataset_train.csv', index_col='Index')
    df_data['Best Hand'].replace({'Left': 0, 'Right': 1}, inplace=True)
    df_plot = df_data.drop(['First Name', 'Last Name', 'Birthday'], axis=1)
    df_melt = df_plot.melt(id_vars="Hogwarts House")
    graph = sns.FacetGrid(df_melt,  # the dataframe to pull from
                      row="Hogwarts House",
                      hue="Hogwarts House",
                      col="variable",
                      aspect=2.5,  # aspect * height = width
                      height=1.5,  # height of each subplot
                      palette=['#4285F4', '#EA4335', '#FBBC05', '#34A853'],  # google colors
                      sharex=False,
                      )
    graph.map(sns.histplot, "value", stat="probability", )
    graph.map(plt.axhline, y=0, lw=4)

    def label(x_val, color, label): # pylint: disable=W0613
        axis = plt.gca()  # get the axes of the current object
        axis.text(0, .2,  # location of text
                label,  # text label
                fontweight="bold", color=color, size=10,  # text attributes
                ha="left", va="center",  # alignment specifications
                transform=axis.transAxes)  # specify axes of transformation

    graph.map(label, "value")
    graph.set_axis_labels("")
    # sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set(style="white")
    graph.set(yticks=[]) #set y ticks to blank
    graph.set_titles('{col_name}')
    graph.despine(bottom=True, left=True)  # remove 'spines'

    for i, _ in enumerate(graph.axes):
        for j, _ in enumerate(graph.axes[i]):
            min_value = min_val(df_data[graph.axes[i][j].title._text].values) # pylint: disable=protected-access
            max_value = max_val(df_data[graph.axes[i][j].title._text].values) # pylint: disable=protected-access
            graph.axes[i, j].set_xlim(min_value - 1, max_value + 1)
    graph.savefig("histogram_all.png")


if __name__ == '__main__':
    histogram()

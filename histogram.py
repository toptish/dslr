import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def histogram():
    df = pd.read_csv('datasets/dataset_train.csv', index_col='Index')
    df_plot = df.drop(['First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
    df_melt = df_plot.melt(id_vars="Hogwarts House")
    g = sns.FacetGrid(df_melt,  # the dataframe to pull from
                      row="Hogwarts House",  # define the column for each subplot row to be differentiated by
                      hue="Hogwarts House",
                      col="variable",  # define the column for each subplot color to be differentiated by
                      aspect=2.5,  # aspect * height = width
                      height=1.5,  # height of each subplot
                      palette=['#4285F4', '#EA4335', '#FBBC05', '#34A853'],  # google colors
                      sharex=False,
                      )
    g.map(sns.histplot, "value", stat="probability", )
    g.map(plt.axhline, y=0, lw=4)

    def label(x, color, label):
        ax = plt.gca()  # get the axes of the current object
        ax.text(0, .2,  # location of text
                label,  # text label
                fontweight="bold", color=color, size=10,  # text attributes
                ha="left", va="center",  # alignment specifications
                transform=ax.transAxes)  # specify axes of transformation

    g.map(label, "value")
    g.set_axis_labels("")
    # sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set(style="white")
    g.set(yticks=[]) #set y ticks to blank
    g.set_titles('{col_name}')
    g.despine(bottom=True, left=True)  # remove 'spines'

    for i in range(len(g.axes)):
        for j in range(len(g.axes[i])):
            min_val = min(df[g.axes[i][j].title._text].values)
            max_val = max(df[g.axes[i][j].title._text].values)
            g.axes[i, j].set_xlim(min_val - 1, max_val + 1)
    fig = g.savefig("histogram_all.png")


if __name__ == '__main__':
    histogram()

import seaborn as sns
import pandas as pd


def pair_plot():
    df = pd.read_csv('datasets/dataset_train.csv', index_col='Index')
    df_plot = df.drop(['First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
    g = sns.pairplot(data=df_plot, hue="Hogwarts House")
    fig = g.savefig("pair_plot.png")


if __name__ == '__main__':
    pair_plot()

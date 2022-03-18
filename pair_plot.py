"""
Piarplot program that shows pair scatter plots for all numeric features of a dataset
"""
import seaborn as sns
import pandas as pd


def pair_plot():
    """
    Creates a pair scatter plot for all numeric features and saves it to .png file
    :return:
    """
    df_data = pd.read_csv('datasets/dataset_train.csv', index_col='Index')
    df_data['Best Hand'].replace({'Left': 0, 'Right': 1}, inplace=True)
    df_plot = df_data.drop(['First Name', 'Last Name', 'Birthday'], axis=1)
    graph = sns.pairplot(data=df_plot, hue="Hogwarts House")
    graph.savefig("pair_plot.png")


if __name__ == '__main__':
    pair_plot()

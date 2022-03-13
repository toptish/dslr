import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from maths import *


def describe():
    """
    Calculates statistic metrics and displays them (like pd.describe())

    """
    df = pd.read_csv('datasets/dataset_train.csv', index_col='Index')
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns.values)
    funcs_map = {'count': count,
                 'mean': mean,
                 'std': std,
                 'min': min,
                 '25%': quartile_25,
                 '50%': quartile_50,
                 '75%': quartile_75,
                 'max': max}
    metrics = list(funcs_map)
    df_describe = pd.DataFrame(columns=numeric_columns, index=metrics)

    for hog_class in numeric_columns:
        for metric in metrics:
            df_describe.loc[metric][hog_class] = funcs_map[metric](df[hog_class].values)
    # print(df_describe)
    pd.set_option('display.max_columns', None)
    pd.options.display.float_format = '{:,.6f}'.format
    print(df_describe)
    # print(df.describe())


if __name__ == '__main__':
    describe()

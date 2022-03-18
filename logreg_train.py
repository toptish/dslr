"""
Program for training
"""
import argparse
import sys
import pandas as pd
import numpy as np
from maths import LogisticRegression as LogReg, min_val, max_val, quartile_50
from messages import Messages


def minmax_normalization(data):
    """

    :param data:
    :return:
    """
    columns = list(data)
    delta = None
    min_ = None
    for col_name in columns:
        delta = max_val(data[col_name]) - min_val(data[col_name])
        min_ = min_val(data[col_name])
        if col_name != "Hogwarts House":
            data[col_name] = data[col_name].apply(lambda x: (x - min_) / delta)


def nan_to_median(data):
    """

    :param data:
    :return:
    """
    columns = list(data)
    for col_name in columns:
        if col_name != "Hogwarts House":
            data[col_name].fillna(quartile_50(data[col_name]), inplace=True)


def parse_args() -> argparse.Namespace:
    """
    Add value to Arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        type=str,
                        help='Path to train data file')
    parser.add_argument('--gradient', '-g',
                        type=str,
                        dest='grad',
                        default='batch',
                        help='Gradient descent method: "batch" (default), '
                             '"mini_batch", "sgd"')
    parser.add_argument('--debug', '-d',
                        dest='debug',
                        action='store_true',
                        default=False,
                        help='Print info about main stages of program')
    # parser.add_argument('--show', '-s',
    # 					action="store_true",
    # 					default=False,
    # 					help='Show the graph of the polynomial function '
    # 						 '//with solutions//')
    return parser.parse_args()


def get_data(path):
    """

    :param path:
    :return:
    """
    data = pd.DataFrame()
    try:
        data = pd.read_csv(path, index_col=0)
    except Exception:
        Messages(f'Cannot read {path}').error_()
        sys.exit(1)

    data["Hogwarts House"] = data["Hogwarts House"].map(
        {"Ravenclaw": 0, "Slytherin": 1, "Gryffindor": 2, "Hufflepuff": 3})
    data.drop(['First Name', 'Last Name', 'Birthday', 'Arithmancy'], axis=1, inplace=True)
    data["Best Hand"].replace({"Right": 0, "Left": 1},
                              inplace=True)
    nan_to_median(data)

    minmax_normalization(data)
    return data


def main():
    """

    :return:
    """
    try:
        args = parse_args()
        if args.grad in ["batch", "mini_batch", "sgd"]:
            pass
        else:
            Messages('Possible Gradient Descent Method is one of ['
                     '"batch", "mini_batch", "sgd"]').error_()
            sys.exit(1)
        val = get_data(args.data)
        val_houses = val.copy()
        val_houses.insert(1, "w_0", 1)
        houses = np.array(
            ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"])
        weights = pd.DataFrame()
        for i in range(0, 4):
            val_houses["Hogwarts House"] = val["Hogwarts House"].apply \
                (lambda x: 1 if x == i else 0)  # pylint: disable=W0640
            logreg = LogReg(grad=args.grad,
                            weights=np.random.uniform(-0.1, 0.1,
                                                      val.shape[1]).reshape(val.shape[1], 1),
                            alpha=1, n_cycle=1000)
            if args.debug is True:
                Messages(f'Training process for "{houses[i]} against all" '
                         f'model with {args.grad} method:').info_()
            logreg.fit(val_houses, args.debug)
            weights[houses[i]] = np.hstack(logreg.weights)

        weights.to_csv("datasets/weights.csv")
        Messages('All done!').ok_()

    except Exception as error:
        Messages(f'{error}').error_()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Program for predicting hogwarts faculties
"""
import argparse
import sys

import pandas as pd
import numpy as np
from logreg_train import get_data
from messages import Messages
from maths import sigmoid


def parse_args() -> argparse.Namespace:
    """
    Add value to Arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        type=str,
                        help='Path to data file')
    parser.add_argument('weights',
                        type=str,
                        help='Path to weights file')
    return parser.parse_args()


def main():
    """
    Main program function
    """
    try:
        args = parse_args()
        val = get_data(args.data)
        weights = pd.read_csv(args.weights, index_col=0)

        val_houses = val.copy()
        val_houses.insert(1, "w_0", 1)
        x_val = val_houses.iloc[:, 1:].values
        houses = np.array(
            ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"])
        predictions = pd.DataFrame()
        for _ in range(0, 4):
            weights_ = weights.iloc[:, _].values
            weights_.shape = (len(weights_), 1)
            prediction = sigmoid(weights_, x_val)
            predictions[houses[_]] = np.hstack(prediction)
        res = predictions.idxmax(axis='columns')
        res_file = pd.DataFrame(res, columns=["Hogwarts House"])
        res_file.index.name = "Index"
        res_file.to_csv("datasets/houses.csv")
        Messages('All done!').ok_()
    except Exception as error:
        Messages(f'{error}').error_()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Math module with essential maths functions
"""
import sys


def mean(data: list) -> float:
    """
    The mean of a data set is the sum of all of the data divided by the size.
    The mean is also known as the average.

    :param data: list of values
    :return: mean value
    """
    return sum_values(data) / count_v(data)


def is_nan(num) -> bool:
    """
    Checks whether value is NaN

    :param num: value to check
    :return: bool
    """
    return num != num


def count_v(data: list) -> float:
    """
    Counts not NaN values in a dataset

    :param data: list of values
    :return: float counting not NaN values
    """
    result = 0.
    for datum in data:
        if not is_nan(datum):
            result += 1
    return result


def sum_values(data: list) -> float:
    """
    Returns the sum of not NaN values

    :param data: list of values
    :return: sum of values excluding NaN
    """
    result = 0.
    for datum in data:
        if not is_nan(datum):
            result += datum
    return result


def min_val(datalist: list) -> float:
    """
    Finds the minimum value for a list of values

    :param datalist: list of values
    :type datalist: list
    :return: min value
    :rtype: float
    """
    min_value = sys.float_info.max
    for number in datalist:
        min_value = number if number < min_value else min_value
    return min_value


def max_val(datalist: list) -> float:
    """
    Finds the maximum value for a list of values

    :param datalist: list of values
    :type datalist: list
    :return: max value
    :rtype: float
    """
    max_value = sys.float_info.min
    for number in datalist:
        max_value = number if number > max_value else max_value
    return max_value


def std(datalist: list) -> float:
    """
    Returns standard deviation for a sample excluding NaNs

    :param datalist: list of values
    :return: standard deviation
    """
    mean_val = mean(datalist)
    datalist = drop_na(datalist)
    sum_sqr_errors = sum_values([(number - mean_val) ** 2 for number in datalist])
    count = count_v(datalist)
    return (sum_sqr_errors / (count - 1)) ** 0.5


def drop_na(datalist: list) -> list:
    """
    Drops not number values from a dataset

    :param datalist: list of values
    :return: list of not NaN values
    """
    return [datum for datum in datalist if not is_nan(datum)]


def percentile(percent: int, dataset: list) -> float:
    """
    Uses the linear interpolation (as default numpy percentile)

    :param percent: required percentile to calculate
    :param dataset: list of data
    :return: percentile
    """
    dataset = sorted(drop_na(dataset))
    count_val = len(dataset)
    break_point = (percent / 100) * (count_val - 1)
    if int(break_point) == break_point:
        return dataset[int(break_point)]
    fraction = break_point - int(break_point)
    left = int(break_point)
    right = left + 1
    return dataset[left] + (dataset[right] - dataset[left]) * fraction


def quartile_25(dataset: list) -> float:
    """
    Calculates 25 quartile (1 quantile) for a dataset
    :param dataset: list of data
    :return: 1 quantile
    """
    return percentile(25, dataset)


def quartile_50(dataset: list) -> float:
    """
    Calculates median for a dataset
    :param dataset: list of data
    :return: 2 quantile (median value)
    """
    return percentile(50, dataset)


def quartile_75(dataset: list) -> float:
    """
    Calculates 25 quartile (1 quantile) for a dataset
    :param dataset: list of data
    :return: 3 quantile
    """
    return percentile(75, dataset)


def var(dataset: list) -> float:
    """
    Variance measures dispersion of data from the mean.
    The formula for variance is the sum of squared differences from
    the mean divided by the size of the data set.

    :param dataset: list of data
    :return: variance
    """
    mean_val = mean(dataset)
    datalist = drop_na(dataset)
    sum_sqr_errors = sum_values([(number - mean_val) ** 2 for number in datalist])
    count = count_v(datalist)
    return sum_sqr_errors / (count - 1)


def sum_of_squares(dataset: list) -> float:
    """
    The sum of squares is the sum of the squared differences between
    data values and the mean.

    :param dataset: list of data
    :return: sum of squares
    """
    mean_val = mean(dataset)
    list_squares = [(value - mean_val) ** 2 for value in dataset]
    return sum_values(list_squares)


def skewness(dataset: list) -> float:
    """
    Skewness[3] describes how far to the left or right a data set distribution
    is distorted from a symmetrical bell curve.
    A distribution with a long left tail is left-skewed, or negatively-skewed.
    A distribution with a long right tail is right-skewed, or positively-skewed.

    :param dataset: list of data
    :return: skewness coef
    """
    dataset = drop_na(dataset)
    count = count_v(dataset)
    mean_val = mean(dataset)
    temp_sum = sum_values([((value - mean_val) / std(dataset)) ** 3 for value in dataset])
    return count / ((count - 1) * (count - 2)) * temp_sum


def kurtosis_42(dataset: list) -> float:
    """
    Excess kurtosis describes the height of the tails of a distribution rather than the extremity
    of the length of the tails.
    Excess kurtosis means that the distribution has a high frequency of data
    outliers.

    :param dataset: list of data
    :return: kurtosis
    """
    dataset = drop_na(dataset)
    count = count_v(dataset)
    mean_val = mean(dataset)
    temp_sum = sum_values([((value - mean_val) / std(dataset)) ** 4 for value in dataset])
    # kurtosis = count * (count + 1) / ((count - 1) * (count - 2) * (count - 3)) * temp_sum
        # 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
    kurtosis = 1 / count * temp_sum - 3
    return kurtosis


def mode_42(dataset: list) -> float:
    """
    The mode is the value or values that occur most frequently in the data set.
    A data set can have more than one mode, and it can also have no mode.

    :param dataset:
    :return:
    """
    dataset = drop_na(dataset)
    dataset = list(dataset)
    return max(set(dataset), key=dataset.count)

import sys


def mean(data: list) -> float:
    """
    Calculates mean for a dataset
    :param data: list of values
    :return: mean value
    """
    return sum_values(data) / count(data)


def isNaN(num) -> bool:
    """
    Checks whether value is NaN

    :param num: value to check
    :return: bool
    """
    return num != num


def count(data: list) -> float:
    """
    Counts not NaN values in a dataset

    :param data: list of values
    :return: float counting not NaN values
    """
    result = 0.
    for datum in data:
        if not isNaN(datum):
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
        if not isNaN(datum):
            result += datum
    return result


def min(datalist: list) -> float:
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


def max(datalist: list) -> float:
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
    Returns standart deviation for a sample excluding NaNs

    :param datalist: list of values
    :return:
    """
    sum_sqr_errors = 0
    count = 0
    mean_val = mean(datalist)
    for number in datalist:
        if not isNaN(number):
            sum_sqr_errors += (number - mean_val) ** 2
            count += 1
    return (sum_sqr_errors / (count - 1)) ** 0.5


def drop_na(datalist: list) -> list:
    """
    Drops not number values from a dataset
    :param datalist: list of values
    :return: list of not NaN values
    """
    return [datum for datum in datalist if not isNaN(datum)]


def quartile(percent: int, dataset: list) -> float:
    """

    :param percent:
    :param dataset:
    :return:
    """
    dataset = sorted(drop_na(dataset))
    count_val = len(dataset) + 1
    breakpoint = (percent / 100) * count_val
#     print(breakpoint)
    if breakpoint.is_integer():
        return dataset[int(breakpoint) - 1]
    else:
        left = int(breakpoint) -1
        right = int(breakpoint)
#         print(left)
#         print(right)
#         print(dataset[left - 1])
#         print(dataset[left])
#         print(dataset[right])
#         print(dataset[right + 1])
        return dataset[left] + (dataset[right] - dataset[left]) * percent / 100


def quartile_25(dataset: list) -> float:
    """

    :param dataset:
    :return:
    """
    return quartile(25, dataset)


def quartile_50(dataset: list) -> float:
    """

    :param dataset:
    :return:
    """
    return quartile(50, dataset)


def quartile_75(dataset: list) -> float:
    """

    :param dataset:
    :return:
    """
    return quartile(75, dataset)

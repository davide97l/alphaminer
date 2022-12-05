from typing import List
import pandas as pd
import numpy as np


def group_return(pred_label: pd.DataFrame = None, reverse: bool = False, N: int = 5, **kwargs) -> pd.DataFrame:
    """
    :param pred_label:
    :param reverse:
    :param N:
    :return:
    """
    if reverse:
        pred_label["score"] *= -1

    pred_label = pred_label.sort_values("score", ascending=False)

    # Group1 ~ Group5 only consider the dropna values
    pred_label_drop = pred_label.dropna(subset=["score"])

    # Group
    t_df = pd.DataFrame(
        {
            "Group%d"
            % (i + 1): pred_label_drop.groupby(level="datetime")["label"].apply(
                lambda x: x[len(x) // N * i : len(x) // N * (i + 1)].mean()  # pylint: disable=W0640
            )
            for i in range(N)
        }
    )
    t_df.index = pd.to_datetime(t_df.index)

    # Long-Short
    t_df["long-short"] = t_df["Group1"] - t_df["Group%d" % N]

    # Long-Average
    t_df["long-average"] = t_df["Group1"] - pred_label.groupby(level="datetime")["label"].mean()

    t_df = t_df.dropna(how="all")  # for days which does not contain label
    t_df.index = t_df.index.strftime("%Y-%m-%d")

    # Cumulative Return By Group
    return t_df


def index_group_return(pred_label: pd.DataFrame = None, reverse: bool = False, N: int = 5, **kwargs) -> tuple:
    t_df = group_return(pred_label, reverse, N, **kwargs)
    mean_df = t_df.mean(axis=0)
    values = mean_df.values
    index_sort = np.argsort(values[:6])
    return mean_df, index_sort


def merge(arr, temp, left, mid, right):
    inv_count = 0

    i = left  # i is index for left subarray*/
    j = mid  # j is index for right subarray*/
    k = left  # k is index for resultant merged subarray*/
    while (i <= mid - 1) and (j <= right):
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            k += 1
            i += 1
        else:
            temp[k] = arr[j]
            k += 1
            j += 1

            # this is tricky -- see above explanation/
            # diagram for merge()*/
            inv_count = inv_count + (mid - i)

    # Copy the remaining elements of left subarray
    # (if there are any) to temp*/
    while i <= mid - 1:
        temp[k] = arr[i]
        k += 1
        i += 1

    # Copy the remaining elements of right subarray
    # (if there are any) to temp*/
    while j <= right:
        temp[k] = arr[j]
        k += 1
        j += 1

    # Copy back the merged elements to original array*/
    for i in range(left, right + 1, 1):
        arr[i] = temp[i]

    return inv_count


# An auxiliary recursive function that sorts the input
# array and returns the number of inversions in the
# array. */
def _mergeSort(arr, temp, left, right):
    inv_count = 0
    if right > left:
        # Divide the array into two parts and call
        # _mergeSortAndCountInv()
        # for each of the parts */
        mid = int((right + left) / 2)

        # Inversion count will be sum of inversions in
        # left-part, right-part and number of inversions
        # in merging */
        inv_count = _mergeSort(arr, temp, left, mid)
        inv_count += _mergeSort(arr, temp, mid + 1, right)

        # Merge the two parts*/
        inv_count += merge(arr, temp, left, mid + 1, right)

    return inv_count


# This function sorts the input array and returns the
# number of inversions in the array */
# taken from https://www.geeksforgeeks.org/number-swaps-sort-adjacent-swapping-allowed/
def count_swaps(arr):
    n = len(arr)
    temp = [0 for i in range(n)]
    return _mergeSort(arr, temp, 0, n - 1)


def cum_ret_groups_corr(seq: List[int]):
    """
    :param seq: list, order of groups.
    :return: correlation of groups order in range (0, 1). 1 when groups are ordered, tend towards zero as groups order
    and correct order are more uncorrelated.
    Groups defined here: https://qlib.readthedocs.io/en/stable/component/report.html#graphical-results.
    The value of this metrics is defined as 1 - s/d where:
    - s is the minimum number of adjacent swaps to order the groups in any order (ascendant or descendant)
    - d is the minimum number of adjacent swaps to order the groups in the worst case (maximum uncorrelated sequence)
    - s <= d, 0 <= s/d <= 1
    """
    assert len(set(seq)) == len(seq), "Input list should not contain repeated values"
    neg_seq = [-x for x in seq]
    count = min(count_swaps(seq.copy()), count_swaps(neg_seq.copy()))
    if len(seq) == 5:  # standard number of groups
        den = 5
    elif len(seq) == 6:  # standard number of groups
        den = 7
    elif len(seq) == 7:  # standard number of groups
        den = 10
    else:
        den = count_swaps(list(range(len(seq), 0, -1))) // 2
        print(den)
    return 1 - count / den


if __name__ == '__main__':
    '''
    group 1..5: 1..5
    long-short: 6
    '''
    seqs = [[1, 2, 3, 4, 5],  # perfect positive correlation
            [5, 4, 3, 2, 1],  # perfect negative correlation
            [1, 2, 3, 5, 4],  # almost perfect positive correlation
            [5, 3, 2, 1, 4],  # low correlation
            [3, 4, 1, 5, 2],  # completely uncorrelated
            [1, 2, 3, 7, 5, 4, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 6, 5, 4],
            [1, 2, 3, 4, 6, 5]
            ]
    for s in seqs:
        print(s, cum_ret_groups_corr(s))

from itertools import combinations
from typing import List

import pandas as pd
import numpy as np

from scipy.special import comb
from sklearn.model_selection import KFold
from .cross_validation import get_train_times

def _get_number_of_backtest_paths(n_train_splits: int, n_test_splits: int) -> float:
    return int(comb(n_train_splits, n_train_splits - n_test_splits) * n_test_splits / n_train_splits)


class CombinatorialPurgedKFold(KFold):
    def __init__(
            self,
            n_splits: int = 3,
            n_test_splits: int = 2,
            samples_info_sets: pd.Series = None,
            pct_embargo: float = 0. ):

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError('The samples_info_sets param must be a pd.Series')
        super(CombinatorialPurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
        self.n_test_splits = n_test_splits
        self.num_backtest_paths = _get_number_of_backtest_paths(self.n_splits, self.n_test_splits)
        self.backtest_paths = []  # Array of backtest paths

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        combinatorial_splits = list(combinations(list(splits_indices.keys()), self.n_test_splits))
        combinatorial_test_ranges = []  # List of test indices formed from combinatorial splits
        for combination in combinatorial_splits:
            temp_test_indices = []  # Array of test indices for current split combination
            for int_index in combination:
                temp_test_indices.append(splits_indices[int_index])
            combinatorial_test_ranges.append(temp_test_indices)
        return combinatorial_test_ranges

    def _fill_backtest_paths(self, train_indices: list, test_splits: list):
        for split in test_splits:
            found = False  # Flag indicating that split was found and filled in one of backtest paths
            for path in self.backtest_paths:
                for path_el in path:
                    if path_el['train'] is None and split == path_el['test'] and found is False:
                        path_el['train'] = np.array(train_indices)
                        path_el['test'] = list(range(split[0], split[-1]))
                        found = True

    # noinspection PyPep8Naming
    def split(
            self,
            X: pd.DataFrame,
            y: pd.Series = None,
            groups = None):
        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        splits_indices = {}
        for index, [start_ix, end_ix] in enumerate(test_ranges):
            splits_indices[index] = [start_ix, end_ix]

        combinatorial_test_ranges = self._generate_combinatorial_test_ranges(splits_indices)
        # Prepare backtest paths
        for _ in range(self.num_backtest_paths):
            path = []
            for split_idx in splits_indices.values():
                path.append({'train': None, 'test': split_idx})
            self.backtest_paths.append(path)

        embargo: int = int(X.shape[0] * self.pct_embargo)
        for test_splits in combinatorial_test_ranges:

            # Embargo
            test_times = pd.Series(
                index=[self.samples_info_sets.iloc[ix[0]] for ix in test_splits],
                data=[
                    self.samples_info_sets.iloc[ix[1] - 1] if ix[1] - 1 + embargo >= X.shape[0] else
                    self.samples_info_sets.iloc[ix[1] - 1 + embargo]
                    for ix in test_splits
                ]
            )

            test_indices = []
            for [start_ix, end_ix] in test_splits:
                test_indices.extend(list(range(start_ix, end_ix)))

            # Purge
            train_times = get_train_times(self.samples_info_sets, test_times)

            # Get indices
            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.samples_info_sets.index.get_loc(train_ix))

            self._fill_backtest_paths(train_indices, test_splits)

            yield np.array(train_indices), np.array(test_indices)
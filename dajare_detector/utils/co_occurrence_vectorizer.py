from typing import List
import itertools
from collections import Counter

import numpy as np


class CoOccurrenceVectorizer:
    """単語セットの共起から情報量"""
    def __init__(self, max_features: int = 0):
        self.max_features = max_features
        self.words = None
        self.matrix = None

    def transform(self, target: List[List[str]]) -> List[float]:
        result = []
        for x, y in target:
            if x in self.words and y in self.words:
                x = self.words.index(x)
                y = self.woords.index(y)
                result.append(self.matrix[x][y])
            else:
                result.append(0)
        return result

    def fit(self, target: List[List[str]]):
        self.words = list(set(itertools.chain.from_iterable(target)))
        if self.max_features > 0:
            self.words = [
                list(k) for k, v in Counter([tuple(sorted(x)) for x in target
                                             ]).most_common(self.max_features)
            ]

        mat = np.zeros((len(self.words), len(self.words)))
        for x, y in target:
            x = self.words.index(x)
            y = self.words.index(y)
            mat[x][y] += 1
            mat[y][x] += 1
        self.matrix = self._calc_ppmi(mat)

    def _calc_ppmi(C, eps=1e-8):
        M = np.zeros_like(C, dtype=np.float32)
        N = np.sum(C)
        S = np.sum(C, axis=0)

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
                M[i, j] = max(0, pmi)
        return M

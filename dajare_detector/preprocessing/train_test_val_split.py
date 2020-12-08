from logging import getLogger

import numpy as np
import gokart
import luigi
from sklearn.model_selection import train_test_split
import swifter  # noqa

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class TrainTestValSplit(DajareTask):
    target = gokart.TaskInstanceParameter()
    __version = luigi.IntParameter(default=1)

    def requires(self):
        return self.target

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)

        col = [x for x in df.columns if x not in {'_id', 'text', 'dajare'}]
        df['feature'] = df.swifter.apply(lambda x: self._explode_cols(x, col),
                                         axis=1).tolist()
        train_features, test_features, train_labels, test_labels = train_test_split(
            df[['_id', 'feature']],
            df['dajare'],
            test_size=0.2,
            stratify=df['dajare'])
        train_features, valid_features, train_labels, valid_labels = train_test_split(
            train_features, train_labels, test_size=0.2, stratify=train_labels)
        self.dump({
            'train_features': train_features,
            'test_features': test_features,
            'valid_features': valid_features,
            'train_labels': train_labels,
            'test_labels': test_labels,
            'valid_labels': valid_labels
        })

    def _explode_cols(self, row, col):
        f = []
        for x in col:
            if type(row[x]) in [int, float]:
                f.append(row[x])
            elif type(row[x]) in [bool]:
                f.append(int(row[x]))
            elif type(row[x]) == np.ndarray:
                f += row[x].tolist()
            elif type(row[x]) == list:
                f += row[x]
        return f

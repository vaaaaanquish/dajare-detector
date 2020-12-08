from logging import getLogger

import gokart
import pandas as pd

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.model.train_lgbm import TrainLGBM

logger = getLogger(__name__)


class PredictLGBM(DajareTask):
    """LGBMによるperdictの結果、正解flag、_idのデータを作成"""
    train_test_val = gokart.TaskInstanceParameter()
    target = gokart.TaskInstanceParameter()

    def requires(self):
        return {
            'model': TrainLGBM(target=self.train_test_val),
            'data': self.train_test_val,
            'target': self.target
        }

    def run(self):
        model = self.load('model')
        data = self.load('data')
        target = self.load('target')[['_id', 'dajare']]

        pred = model.predict(data['test_features']['feature'].tolist(),
                             num_iteration=model.best_iteration)
        df = pd.DataFrame({
            'pred': pred,
            '_id': data['test_features']['_id'].tolist()
        })
        df = pd.merge(df, target, on='_id')
        self.dump(df)

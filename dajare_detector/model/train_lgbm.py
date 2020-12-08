from logging import getLogger

import gokart
import optuna.integration.lightgbm as lgb
from lightgbm import Dataset

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class TrainLGBM(DajareTask):
    target = gokart.TaskInstanceParameter()

    def requires(self):
        return self.target

    def run(self):
        data = self.load()

        trains = Dataset(data['train_features']['feature'].tolist(),
                         label=data['train_labels'].tolist())
        valids = Dataset(data['valid_features']['feature'].tolist(),
                         data['valid_labels'].tolist())

        params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1}
        model = lgb.train(params,
                          trains,
                          verbose_eval=False,
                          valid_sets=valids,
                          num_boost_round=200,
                          early_stopping_rounds=5)
        self.dump(model)

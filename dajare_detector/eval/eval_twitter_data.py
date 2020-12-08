from logging import getLogger

import pandas as pd
import luigi

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.predict.predict_twitter_data import PredictTwitterData
from dajare_detector.make_data.make_twitter_dataset import MakeTwitterDataset

logger = getLogger(__name__)


class EvalTwitterData(DajareTask):
    file_path = luigi.Parameter()

    def requires(self):
        return {
            'pred': PredictTwitterData(file_path=self.file_path),
            'origin': MakeTwitterDataset(file_path=self.file_path)
        }

    def run(self):
        origin = self.load_data_frame('origin')
        df = self.load_data_frame('pred').reset_index(drop=True)
        df = df[df['pred'].apply(lambda x: 1 if x >= 0.5 else 0)]
        df = pd.merge(df, origin, on='_id', how='left')
        self.dump(df)

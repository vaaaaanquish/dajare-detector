from logging import getLogger

import gokart
import pandas as pd

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.featurize.make_bert_feature import MakeBertFeature

logger = getLogger(__name__)


class MakeBertFeatureDataset(DajareTask):
    """BERT特徴量のみのデータセット生成"""
    target = gokart.TaskInstanceParameter()

    def requires(self):
        return {
            'vector': MakeBertFeature(target=self.target),
            'text': self.target,
        }

    def run(self):
        vector_df = self.load_data_frame('vector').reset_index(drop=True)
        text_df = self.load_data_frame('text').reset_index(drop=True)
        df = pd.merge(vector_df, text_df, on='_id')
        self.dump(df[['_id', 'text', 'bert_vector', 'dajare']])

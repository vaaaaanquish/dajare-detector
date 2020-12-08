from logging import getLogger

import pandas as pd

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.make_data.load_dajare_data import LoadDajareData
from dajare_detector.make_data.load_aozora_data import LoadAozoraData

logger = getLogger(__name__)


class MakeDataset(DajareTask):
    """基準となる青空文庫+ダジャレのデータを生成"""
    def requires(self):
        return {
            'dajare_text': self.clone(LoadDajareData),
            'aozora_text': self.clone(LoadAozoraData)
        }

    def run(self):
        dajare_text_df = self.load_data_frame('dajare_text')
        aozora_text_df = self.load_data_frame('aozora_text')
        dajare_text_df['dajare'] = 1
        aozora_text_df['dajare'] = 0
        df = pd.concat([dajare_text_df, aozora_text_df]).reset_index(drop=True)
        df['_id'] = pd.Series(list(range(len(df))))  # reindex
        self.dump(df[['_id', 'text', 'dajare']])

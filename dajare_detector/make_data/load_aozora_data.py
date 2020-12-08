from logging import getLogger

import pandas as pd

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class LoadAozoraData(DajareTask):
    def run(self):
        df = pd.read_csv(
            'https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip'
        )
        df['text'] = df['作品名']
        df['_id'] = df['作品ID']
        self.dump(df[['_id', 'text']])

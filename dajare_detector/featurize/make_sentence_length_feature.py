from logging import getLogger

import gokart

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class MakeSentenceLengthFeature(DajareTask):
    """文字列の長さ"""
    target = gokart.TaskInstanceParameter()

    def requires(self):
        return self.target

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        df['sentence_length'] = df['text'].str.len()
        self.dump(df[['_id', 'sentence_length']])

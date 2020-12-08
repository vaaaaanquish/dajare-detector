from logging import getLogger

import gokart
import swifter  # noqa

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.preprocessing.tokenize_data import TokenizeData

logger = getLogger(__name__)


class MakeWordCountFeature(DajareTask):
    """単語数"""
    target = gokart.TaskInstanceParameter()

    def requires(self):
        return TokenizeData(target=self.target)

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        df['word_count'] = df['tokenized_text'].swifter.apply(len)
        self.dump(df[['_id', 'word_count']])

from logging import getLogger

import luigi
import swifter  # noqa
import gokart

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class MakeSplitedPattern(DajareTask):
    """カタカナの繰り返し列を生成."""
    split_window_size = luigi.IntParameter()
    target = gokart.TaskInstanceParameter()

    def requires(self):
        return self.target

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        df['splited_pattern'] = df['normalized_kana_pattern'].swifter.apply(
            self._devide_list)
        self.dump(df[['_id', 'splited_pattern']])

    def _devide_list(self, sentences):
        return [self._devide(s.replace(' ', '')) for s in sentences]

    def _devide(self, sentence):
        repeat_num = len(sentence) - (self.split_window_size - 1)
        return [
            sentence[i:i + self.split_window_size] for i in range(repeat_num)
        ]

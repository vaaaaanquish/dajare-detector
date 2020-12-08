from logging import getLogger

import gokart
import luigi
import swifter  # noqa

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.preprocessing.make_kana_pattern import MakeKanaPattern
from dajare_detector.preprocessing.make_splited_pattern import MakeSplitedPattern
from dajare_detector.preprocessing.decide_kana_pattern import DecideKanaPattern
from dajare_detector.preprocessing.normalize_kana_pattern import NormalizeKanaPattern

logger = getLogger(__name__)


class MakeDecideKanaFeature(DajareTask):
    """カタカナの繰り返しが発生したか"""
    target = gokart.TaskInstanceParameter()
    split_window_size = luigi.IntParameter()

    def requires(self):
        kana_task = NormalizeKanaPattern(target=MakeKanaPattern(
            target=self.target))
        split_task = MakeSplitedPattern(
            target=kana_task, split_window_size=self.split_window_size)
        return DecideKanaPattern(split_pattern_target=split_task,
                                 kana_pattern_target=kana_task,
                                 split_window_size=self.split_window_size)

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        df[f'decide_kana_{self.split_window_size}'] = df[
            'decide_kana_flag_list'].swifter.apply(lambda x: 1
                                                   if any(x) else 0)
        self.dump(df[['_id', f'decide_kana_{self.split_window_size}']])

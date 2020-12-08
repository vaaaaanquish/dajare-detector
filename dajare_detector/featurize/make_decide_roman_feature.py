from logging import getLogger

import gokart
import luigi
import swifter  # noqa

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.preprocessing.make_roman_pattern import MakeRomanPattern
from dajare_detector.preprocessing.make_kana_pattern import MakeKanaPattern
from dajare_detector.preprocessing.decide_roman_pattern import DecideRomanPattern
from dajare_detector.preprocessing.normalize_kana_pattern import NormalizeKanaPattern

logger = getLogger(__name__)


class MakeDecideRomanFeature(DajareTask):
    """romanの繰り返しが発生したか"""
    target = gokart.TaskInstanceParameter()
    min_roman = luigi.IntParameter()

    def requires(self):
        return DecideRomanPattern(
            min_roman=self.min_roman,
            target=MakeRomanPattern(target=NormalizeKanaPattern(
                target=MakeKanaPattern(target=self.target))))

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        df[f'decide_roman_{self.min_roman}'] = df[
            'decide_roman_flag_list'].swifter.apply(lambda x: 1
                                                    if any(x) else 0)
        self.dump(df[['_id', f'decide_roman_{self.min_roman}']])
